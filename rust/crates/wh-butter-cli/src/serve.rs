// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible HTTP/1.1 server (std::net). One loaded GGUF,
//! Chat Completions over `wh_butter::generate`. Prefix reuse is real
//! on append-only prompts; rewind / fork rebuilds because hybrid
//! recurrent GDN/conv state cannot rewind.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use wh_butter::generate::{watch_disconnect, GenerateOpts, LiveGenerateCache, LoadedModel};

pub use wh_butter::generate::{common_prefix, n_past};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeFlags {
    model: Option<String>,
    port: u16,
    host: String,
    max_context: Option<usize>,
    enable_thinking: bool,
    kv_cache: Option<String>,
}

fn parse_serve_flags(args: &[String]) -> Result<ServeFlags, String> {
    let mut flags = ServeFlags {
        model: None,
        port: 8000,
        host: "127.0.0.1".to_string(),
        max_context: None,
        enable_thinking: false,
        kv_cache: None,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                flags.model = args.get(i).cloned();
            }
            "--port" => {
                i += 1;
                flags.port = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--port needs a u16".to_string())?;
            }
            "--host" => {
                i += 1;
                flags.host = args.get(i).cloned().ok_or_else(|| "--host needs an address".to_string())?;
            }
            "--max-context" => {
                i += 1;
                let n: usize = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--max-context needs a positive integer".to_string())?;
                if n == 0 {
                    return Err("--max-context must be > 0".into());
                }
                flags.max_context = Some(n);
            }
            "--kv-cache" => {
                i += 1;
                flags.kv_cache = args.get(i).cloned();
            }
            "--aura-decode-path" | "--kv-window-size" | "--kv-window-keep"
            | "--reasoning-effort" => {
                i += 1;
                if args.get(i).is_none() {
                    return Err(format!("{} needs a value", args[i - 1]));
                }
            }
            "--enable-thinking" => {
                flags.enable_thinking = true;
            }
            other => return Err(format!("unrecognized flag {other:?}")),
        }
        i += 1;
    }
    Ok(flags)
}

pub fn run(args: &[String]) {
    let flags = match parse_serve_flags(args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("butter serve: {e}");
            std::process::exit(2);
        }
    };
    let ServeFlags { model, port, host, max_context, enable_thinking, kv_cache } = flags;
    if let Some(kind) = kv_cache.as_deref() {
        let k = kind.to_ascii_lowercase();
        if k != "raw" {
            // TODO: affine / AURA KV on CUDA/Vulkan : Apple path only today.
            eprintln!(
                "butter serve: --kv-cache {kind:?} is Apple-only; this backend uses raw KV"
            );
        }
    }
    let Some(model_path) = model else {
        eprintln!("butter serve: --model <path> is required");
        std::process::exit(2);
    };

    let devices = wh_butter::devices();
    let Some(dev) = devices.first() else {
        eprintln!(
            "butter serve: no live device (compiled backends: {:?})",
            wh_butter::compiled_backends()
        );
        std::process::exit(1);
    };
    eprintln!("butter serve: loading {model_path} on [{}] {}", dev.backend().as_str(), dev.name());
    let loaded = match wh_butter::generate::load(dev.as_ref(), &model_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("butter serve: {e}");
            std::process::exit(1);
        }
    };
    let model_id = model_path.clone();
    let state = Arc::new(ServerState {
        dev: dev.clone(),
        loaded: Mutex::new(loaded),
        model_id: Mutex::new(model_id),
        live: Mutex::new(None),
        current_abort: Mutex::new(Arc::new(AtomicBool::new(false))),
        load_count: AtomicUsize::new(1),
        max_context,
        enable_thinking,
    });

    let bind = format!("{host}:{port}");
    let listener = match TcpListener::bind(&bind) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("butter serve: bind {bind}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("butter serve listening on http://{bind}");
    eprintln!("  GET  /health");
    eprintln!("  GET  /v1/models");
    eprintln!("  POST /v1/chat/completions");
    eprintln!("  POST /v1/messages");
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let state = state.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle(&state, stream) {
                        eprintln!("butter serve: {e}");
                    }
                });
            }
            Err(e) => eprintln!("butter serve: accept: {e}"),
        }
    }
}

struct ServerState {
    dev: std::sync::Arc<dyn wh_butter::Device>,
    loaded: Mutex<LoadedModel>,
    model_id: Mutex<String>,
    live: Mutex<Option<LiveGenerateCache>>,
    current_abort: Mutex<Arc<AtomicBool>>,
    load_count: AtomicUsize,
    max_context: Option<usize>,
    enable_thinking: bool,
}

fn ensure_resident(state: &ServerState, want: &str) -> Result<(), String> {
    if want.is_empty() {
        return Ok(());
    }
    if *state.model_id.lock().unwrap() == want {
        return Ok(());
    }
    let loaded = wh_butter::generate::load(state.dev.as_ref(), want).map_err(|e| e.to_string())?;
    state.current_abort.lock().unwrap().store(true, Ordering::Relaxed);
    *state.loaded.lock().unwrap() = loaded;
    *state.model_id.lock().unwrap() = want.to_string();
    *state.live.lock().unwrap() = None;
    state.load_count.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn handle(state: &ServerState, mut stream: TcpStream) -> std::io::Result<()> {
    let mut buf = vec![0u8; 4096];
    let mut acc = Vec::new();
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        acc.extend_from_slice(&buf[..n]);
        if let Some(header_end) = find_header_end(&acc) {
            let header = String::from_utf8_lossy(&acc[..header_end]);
            let (method, path, content_length) = parse_request_line(&header);
            while acc.len() < header_end + content_length {
                let n = stream.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                acc.extend_from_slice(&buf[..n]);
            }
            let body = acc.get(header_end..header_end + content_length).unwrap_or(&[]).to_vec();
            return dispatch(state, &mut stream, &method, &path, &body);
        }
        if acc.len() > 64 * 1024 * 1024 {
            return write_response(&mut stream, 413, "text/plain", b"payload too large");
        }
    }
}

fn dispatch(
    state: &ServerState,
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let path = path.split('?').next().unwrap_or(path);
    if method == "OPTIONS" {
        return write_response(stream, 204, "text/plain", b"");
    }
    match (method, path) {
        ("GET", "/health") => write_response(stream, 200, "text/plain", b"ok\n"),
        ("GET", "/v1/models") => {
            let id = state.model_id.lock().unwrap().clone();
            let mut entry = json!({
                "id": id,
                "object": "model",
                "owned_by": "butter"
            });
            if let Some(n) = state.max_context {
                entry["max_model_len"] = json!(n);
                entry["context_length"] = json!(n);
            }
            let payload = json!({
                "object": "list",
                "data": [entry]
            });
            write_response(stream, 200, "application/json", payload.to_string().as_bytes())
        }
        ("POST", "/v1/chat/completions") => chat_completions(state, stream, body),
        ("POST", "/v1/messages") | ("POST", "/v1/messages/") => messages(state, stream, body),
        _ => write_response(stream, 404, "text/plain", b"not found"),
    }
}

fn chat_completions(state: &ServerState, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let v: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            let err = json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}});
            return write_response(stream, 400, "application/json", err.to_string().as_bytes());
        }
    };
    let messages = match v.get("messages").and_then(|m| m.as_array()) {
        Some(m) if !m.is_empty() => m,
        _ => {
            let err = json!({"error": {"message": "messages required", "type": "invalid_request_error"}});
            return write_response(stream, 400, "application/json", err.to_string().as_bytes());
        }
    };
    let kwargs_thinking = v
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|b| b.as_bool());
    let thinking = kwargs_thinking.unwrap_or(state.enable_thinking);
    let prompt = render_messages(messages, thinking);
    let stream_flag = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let include_usage = v
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|b| b.as_bool())
        .unwrap_or(false);
    let max_new = v
        .get("max_tokens")
        .or_else(|| v.get("max_completion_tokens"))
        .and_then(|n| n.as_u64())
        .unwrap_or(256) as usize;
    let requested = v.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
    let model_id = if requested.is_empty() {
        state.model_id.lock().unwrap().clone()
    } else {
        requested
    };
    if let Err(e) = ensure_resident(state, &model_id) {
        let err = json!({"error": {"message": e, "type": "invalid_request_error"}});
        return write_response(stream, 404, "application/json", err.to_string().as_bytes());
    }
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let id = format!("chatcmpl-{}", created);
    let abort = watch_disconnect(stream);
    *state.current_abort.lock().unwrap() = abort.clone();

    let opts = GenerateOpts {
        max_new,
        chunk: 0,
        sampling: wh_butter::generate::Sampling::Greedy,
        seed: v.get("seed").and_then(|s| s.as_u64()).unwrap_or(0),
        stream: false,
        max_context: state.max_context,
        abort: Some(abort.clone()),
    };

    if stream_flag {
        write_sse_headers(stream)?;
        let id_c = id.clone();
        let model_c = model_id.to_string();
        let loaded = state.loaded.lock().unwrap();
        let mut live = state.live.lock().unwrap();
        let report = match wh_butter::generate::generate_with_sink_cached(
            state.dev.as_ref(),
            &loaded,
            &prompt,
            &opts,
            &mut live,
            |text, _tok| {
                let chunk = json!({
                    "id": id_c,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_c,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": Value::Null
                    }]
                });
                write_sse(stream, &chunk.to_string()).is_ok()
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                let err = json!({"error": {"message": e.to_string()}});
                let _ = write_sse(stream, &err.to_string());
                let _ = write_sse_done(stream);
                return Ok(());
            }
        };
        drop(live);
        abort.store(true, Ordering::Relaxed);
        let tools = recover_tools(&report.text);
        let finish_reason = if tools.is_some() { "tool_calls" } else { "stop" };
        if let Some((name, args, _)) = &tools {
            let tc = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "chatcmpl-tool-recovered-000000000000",
                        "type": "function",
                        "function": {"name": name, "arguments": args}
                    }]},
                    "finish_reason": Value::Null
                }]
            });
            write_sse(stream, &tc.to_string())?;
        }
        let finish = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        write_sse(stream, &finish.to_string())?;
        if include_usage {
            let usage = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": report.prompt_tokens,
                    "completion_tokens": report.generated_tokens,
                    "total_tokens": report.prompt_tokens + report.generated_tokens,
                    "cached_tokens": report.cached_tokens
                }
            });
            write_sse(stream, &usage.to_string())?;
        }
        write_sse_done(stream)?;
        Ok(())
    } else {
        let loaded = state.loaded.lock().unwrap();
        let mut live = state.live.lock().unwrap();
        let report = match wh_butter::generate::generate_with_sink_cached(
            state.dev.as_ref(),
            &loaded,
            &prompt,
            &opts,
            &mut live,
            |_, _| true,
        ) {
            Ok(r) => r,
            Err(e) => {
                let err = json!({"error": {"message": e.to_string(), "type": "server_error"}});
                return write_response(stream, 500, "application/json", err.to_string().as_bytes());
            }
        };
        drop(live);
        abort.store(true, Ordering::Relaxed);
        let tools = recover_tools(&report.text);
        let (content, tool_calls, finish_reason) = match &tools {
            Some((name, args, residual)) => (
                if residual.is_empty() { Value::Null } else { json!(residual) },
                json!([{
                    "id": "chatcmpl-tool-recovered-000000000000",
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                }]),
                "tool_calls",
            ),
            None => (json!(report.text), Value::Null, "stop"),
        };
        let mut message = json!({"role": "assistant", "content": content});
        if !tool_calls.is_null() {
            message["tool_calls"] = tool_calls;
        }
        let payload = json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": report.prompt_tokens,
                "completion_tokens": report.generated_tokens,
                "total_tokens": report.prompt_tokens + report.generated_tokens,
                "cached_tokens": report.cached_tokens
            }
        });
        write_response(stream, 200, "application/json", payload.to_string().as_bytes())
    }
}

#[derive(Debug)]
struct ParsedMessages {
    max_new: usize,
    stream: bool,
    model: Option<String>,
    seed: u64,
    thinking: Option<bool>,
    prompt: String,
    flat: Vec<Value>,
}

/// Codec entry: 400s a Messages client would see, without touching the GPU.
fn parse_messages_body(body: &[u8]) -> Result<ParsedMessages, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    if v.get("max_tokens").is_none() {
        return Err("max_tokens required".into());
    }
    let max_new = v.get("max_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
    let flat = flatten_anthropic(&v).map_err(|_| "messages required".to_string())?;
    if flat.is_empty() {
        return Err("messages required".into());
    }
    let thinking = v
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|s| s.as_str())
        .map(|s| s == "enabled");
    Ok(ParsedMessages {
        max_new,
        stream: v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
        model: v.get("model").and_then(|m| m.as_str()).map(|s| s.to_string()),
        seed: v.get("seed").and_then(|s| s.as_u64()).unwrap_or(0),
        thinking,
        prompt: render_messages(&flat, thinking.unwrap_or(false)),
        flat,
    })
}

fn event_frame(event: &str, json: &str) -> String {
    format!("event: {event}\r\ndata: {json}\r\n\r\n")
}

fn messages_stop_reason(assembled: &str, generated: usize, max_new: usize) -> &'static str {
    if recover_tools(assembled).is_some() {
        "tool_use"
    } else if max_new > 0 && generated >= max_new {
        "max_tokens"
    } else {
        "end_turn"
    }
}

/// Post-decode Anthropic events (what a Messages client requires after text).
fn messages_finish_events(assembled: &str, text_started: bool, generated: usize, max_new: usize) -> String {
    let mut out = String::new();
    let mut started = text_started;
    let stop = messages_stop_reason(assembled, generated, max_new);
    if let Some((name, args, residual)) = recover_tools(assembled) {
        if !residual.is_empty() {
            if !started {
                out.push_str(&event_frame(
                    "content_block_start",
                    &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string(),
                ));
                started = true;
            }
            out.push_str(&event_frame(
                "content_block_delta",
                &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":residual}}).to_string(),
            ));
        }
        if started {
            out.push_str(&event_frame(
                "content_block_stop",
                &json!({"type":"content_block_stop","index":0}).to_string(),
            ));
        }
        let idx = if started { 1 } else { 0 };
        out.push_str(&event_frame(
            "content_block_start",
            &json!({"type":"content_block_start","index":idx,"content_block":{"type":"tool_use","id":"toolu_000000000000","name":name,"input":{}}}).to_string(),
        ));
        out.push_str(&event_frame(
            "content_block_delta",
            &json!({"type":"content_block_delta","index":idx,"delta":{"type":"input_json_delta","partial_json":args}}).to_string(),
        ));
        out.push_str(&event_frame(
            "content_block_stop",
            &json!({"type":"content_block_stop","index":idx}).to_string(),
        ));
    } else {
        if !started {
            out.push_str(&event_frame(
                "content_block_start",
                &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string(),
            ));
        }
        out.push_str(&event_frame(
            "content_block_stop",
            &json!({"type":"content_block_stop","index":0}).to_string(),
        ));
    }
    out.push_str(&event_frame(
        "message_delta",
        &json!({
            "type":"message_delta",
            "delta":{"stop_reason":stop,"stop_sequence":Value::Null},
            "usage":{"output_tokens": generated}
        })
        .to_string(),
    ));
    out.push_str(&event_frame("message_stop", &json!({"type":"message_stop"}).to_string()));
    out
}

fn messages(state: &ServerState, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let parsed = match parse_messages_body(body) {
        Ok(p) => p,
        Err(msg) => {
            let err = json!({"error": {"message": msg, "type": "invalid_request_error"}});
            return write_response(stream, 400, "application/json", err.to_string().as_bytes());
        }
    };
    let max_new = parsed.max_new;
    let stream_flag = parsed.stream;
    let thinking = parsed.thinking.unwrap_or(state.enable_thinking);
    let rendered = render_messages(&parsed.flat, thinking);
    let model_id = parsed
        .model
        .clone()
        .unwrap_or_else(|| state.model_id.lock().unwrap().clone());
    if let Err(e) = ensure_resident(state, &model_id) {
        let err = json!({"error": {"message": e, "type": "invalid_request_error"}});
        return write_response(stream, 404, "application/json", err.to_string().as_bytes());
    }
    let id = format!(
        "msg_{}",
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    );
    let abort = watch_disconnect(stream);
    *state.current_abort.lock().unwrap() = abort.clone();
    let opts = GenerateOpts {
        max_new,
        chunk: 0,
        sampling: wh_butter::generate::Sampling::Greedy,
        seed: parsed.seed,
        stream: false,
        max_context: state.max_context,
        abort: Some(abort.clone()),
    };

    if stream_flag {
        write_event_headers(stream)?;
        write_event(
            stream,
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_id,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            })
            .to_string(),
        )?;
        let mut assembled = String::new();
        let mut text_started = false;
        let loaded = state.loaded.lock().unwrap();
        let mut live = state.live.lock().unwrap();
        let report = match wh_butter::generate::generate_with_sink_cached(
            state.dev.as_ref(),
            &loaded,
            &rendered,
            &opts,
            &mut live,
            |text, _tok| {
                assembled.push_str(text);
                if assembled.contains("<tool_call") || assembled.contains("[TOOL_CALLS]") {
                    return true;
                }
                if !text_started {
                    let _ = write_event(
                        stream,
                        "content_block_start",
                        &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string(),
                    );
                    text_started = true;
                }
                write_event(
                    stream,
                    "content_block_delta",
                    &json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}).to_string(),
                )
                .is_ok()
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                let _ = write_event(
                    stream,
                    "error",
                    &json!({"type":"error","error":{"message": e.to_string()}}).to_string(),
                );
                let _ = write_event(stream, "message_stop", &json!({"type":"message_stop"}).to_string());
                return Ok(());
            }
        };
        drop(live);
        abort.store(true, Ordering::Relaxed);
        stream.write_all(
            messages_finish_events(&assembled, text_started, report.generated_tokens, max_new)
                .as_bytes(),
        )?;
        stream.flush()?;
        Ok(())
    } else {
        let loaded = state.loaded.lock().unwrap();
        let mut live = state.live.lock().unwrap();
        let report = match wh_butter::generate::generate_with_sink_cached(
            state.dev.as_ref(),
            &loaded,
            &rendered,
            &opts,
            &mut live,
            |_, _| true,
        ) {
            Ok(r) => r,
            Err(e) => {
                let err = json!({"error": {"message": e.to_string(), "type": "server_error"}});
                return write_response(stream, 500, "application/json", err.to_string().as_bytes());
            }
        };
        drop(live);
        abort.store(true, Ordering::Relaxed);
        let mut finish = if max_new > 0 && report.generated_tokens >= max_new {
            "max_tokens"
        } else {
            "end_turn"
        };
        let mut content: Vec<Value> = Vec::new();
        if let Some((name, args, residual)) = recover_tools(&report.text) {
            if !residual.is_empty() {
                content.push(json!({"type":"text","text":residual}));
            }
            let input: Value = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
            content.push(json!({"type":"tool_use","id":"toolu_000000000000","name":name,"input":input}));
            finish = "tool_use";
        } else {
            content.push(json!({"type":"text","text": report.text}));
        }
        let payload = json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": model_id,
            "stop_reason": finish,
            "stop_sequence": Value::Null,
            "usage": {
                "input_tokens": report.prompt_tokens,
                "output_tokens": report.generated_tokens
            }
        });
        write_response(stream, 200, "application/json", payload.to_string().as_bytes())
    }
}

fn flatten_anthropic(v: &Value) -> Result<Vec<Value>, ()> {
    let mut out = Vec::new();
    if let Some(sys) = v.get("system") {
        let text = extract_content(Some(sys));
        if !text.is_empty() {
            out.push(json!({"role":"system","content":text}));
        }
    }
    let Some(messages) = v.get("messages").and_then(|m| m.as_array()) else {
        return Err(());
    };
    if messages.is_empty() && out.is_empty() {
        return Err(());
    }
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        match m.get("content") {
            Some(Value::String(s)) => {
                let role = if role == "assistant" { "assistant" } else if role == "tool" { "tool" } else { "user" };
                out.push(json!({"role":role,"content":s}));
            }
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for p in parts {
                    if let Some(t) = p.as_str() {
                        text.push_str(t);
                        continue;
                    }
                    let Some(d) = p.as_object() else { continue };
                    let ty = d.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                    match ty {
                        "text" => {
                            if let Some(t) = d.get("text").and_then(|t| t.as_str()) {
                                text.push_str(t);
                            }
                        }
                        "tool_result" => {
                            if !text.is_empty() {
                                out.push(json!({"role":"user","content":text}));
                                text.clear();
                            }
                            out.push(json!({"role":"tool","content": extract_content(d.get("content"))}));
                        }
                        "tool_use" => {
                            if !text.is_empty() {
                                out.push(json!({"role":"assistant","content":text}));
                                text.clear();
                            }
                            let name = d.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let input = d.get("input").cloned().unwrap_or(json!({}));
                            out.push(json!({"role":"assistant","content": format!("{name} {input}")}));
                        }
                        _ => {
                            if let Some(t) = d.get("text").and_then(|t| t.as_str()) {
                                text.push_str(t);
                            }
                        }
                    }
                }
                if !text.is_empty() {
                    let role = if role == "assistant" { "assistant" } else { "user" };
                    out.push(json!({"role":role,"content":text}));
                }
            }
            _ => {}
        }
    }
    if out.is_empty() { Err(()) } else { Ok(out) }
}

fn strip_think_blocks(text: &str) -> String {
    let mut s = text.to_string();
    while let Some(start) = s.find("<think>") {
        if let Some(rel_end) = s[start..].find("</think>") {
            let end = start + rel_end + "</think>".len();
            s.replace_range(start..end, "");
        } else {
            s.truncate(start);
            break;
        }
    }
    s
}

fn recover_tools(text: &str) -> Option<(String, String, String)> {
    let visible = strip_think_blocks(text);
    recover_json_markup(&visible).or_else(|| recover_xml_markup(&visible))
}

/// JSON tool-call markup leak -> (name, arguments JSON string, residual).
fn recover_json_markup(text: &str) -> Option<(String, String, String)> {
    let start = text.find("<tool_call>")?;
    let end = text.find("</tool_call>")?;
    if end <= start {
        return None;
    }
    let inner = text[start + "<tool_call>".len()..end].trim();
    if inner.contains("<function=") {
        return None;
    }
    let obj: Value = serde_json::from_str(inner).ok()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let args = match obj.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => "{}".into(),
    };
    let residual = format!("{}{}", &text[..start], &text[end + "</tool_call>".len()..])
        .trim()
        .to_string();
    Some((name, args, residual))
}

fn recover_xml_markup(text: &str) -> Option<(String, String, String)> {
    let start = text.find("<tool_call>")?;
    let end = text.find("</tool_call>")?;
    if end <= start {
        return None;
    }
    let inner = text[start + "<tool_call>".len()..end].trim();
    let fname = inner.find("<function=")?;
    let name_start = fname + "<function=".len();
    let name_end = inner[name_start..].find('>')?;
    let name = inner[name_start..name_start + name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut rest = &inner[name_start + name_end + 1..];
    while let Some(p) = rest.find("<parameter=") {
        rest = &rest[p + "<parameter=".len()..];
        let gt = rest.find('>')?;
        let key = rest[..gt].trim().to_string();
        rest = &rest[gt + 1..];
        let close = rest.find("</parameter>")?;
        let val = rest[..close].trim().to_string();
        args.insert(key, json!(val));
        rest = &rest[close + "</parameter>".len()..];
    }
    let residual = format!("{}{}", &text[..start], &text[end + "</tool_call>".len()..])
        .trim()
        .to_string();
    Some((name, Value::Object(args).to_string(), residual))
}

fn write_event(stream: &mut TcpStream, event: &str, json: &str) -> std::io::Result<()> {
    stream.write_all(event_frame(event, json).as_bytes())?;
    stream.flush()
}

fn render_messages(messages: &[Value], enable_thinking: bool) -> String {
    let mut s = String::new();
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let role = if role == "developer" { "system" } else { role };
        let content = extract_content(m.get("content"));
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    // Qwen 3.x chat template: thinking off closes the think block up
    // front so the model doesn't emit a reasoning prefix.
    if enable_thinking {
        s.push_str("<|im_start|>assistant\n");
    } else {
        s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    s
}

fn extract_content(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|p| {
                p.as_str().map(|s| s.to_string()).or_else(|| {
                    p.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                })
            })
            .collect(),
        _ => String::new(),
    }
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

fn parse_request_line(header: &str) -> (String, String, usize) {
    let mut method = "GET".to_string();
    let mut path = "/".to_string();
    let mut content_length = 0usize;
    for (i, line) in header.split("\r\n").enumerate() {
        if i == 0 {
            let mut it = line.split(' ');
            method = it.next().unwrap_or("GET").to_string();
            path = it.next().unwrap_or("/").to_string();
        } else if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    (method, path, content_length)
}

fn cors_headers() -> &'static str {
    "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
}

fn write_response(stream: &mut TcpStream, status: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
        body.len(),
        cors_headers()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n{}\r\n",
        cors_headers()
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn write_event_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n{}\r\n",
        cors_headers()
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn write_sse(stream: &mut TcpStream, json: &str) -> std::io::Result<()> {
    stream.write_all(format!("data: {json}\r\n\r\n").as_bytes())?;
    stream.flush()
}

fn write_sse_done(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"data: [DONE]\r\n\r\n")?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_past_decrements_when_equal() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(n_past(&[1, 2, 3], &[1, 2, 3]), 2);
        assert_eq!(n_past(&[1, 2, 3, 4], &[1, 2, 3]), 2);
        assert_eq!(n_past(&[], &[1]), 0);
        assert_eq!(n_past(&[1], &[]), 0);
    }

    // Messages codec (no GPU).

    #[test]
    fn messages_missing_max_tokens_is_err() {
        // Mutation: accept the body without max_tokens.
        let err = parse_messages_body(
            br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("max_tokens"));
    }

    #[test]
    fn messages_empty_is_err() {
        let err = parse_messages_body(br#"{"max_tokens":32,"messages":[]}"#).unwrap_err();
        assert!(err.contains("messages"));
    }

    #[test]
    fn messages_unknown_fields_still_parse() {
        // Mutation: 400 on thinking/metadata/cache_control.
        let p = parse_messages_body(
            br#"{"model":"x","max_tokens":32,"thinking":{"type":"enabled"},"metadata":{"user_id":"u"},"cache_control":{"type":"ephemeral"},"tool_choice":"auto","strict":true,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(p.max_new, 32);
        assert!(p.prompt.contains("hi"));
        assert_eq!(p.thinking, Some(true));
        assert!(!p.prompt.contains("</think>"));
    }

    #[test]
    fn serve_flags_max_context_and_thinking() {
        // Mutation: unrecognized flag exit 2.
        let f = parse_serve_flags(&[
            "--model".into(),
            "m.gguf".into(),
            "--max-context".into(),
            "8192".into(),
            "--kv-cache".into(),
            "aura".into(),
            "--enable-thinking".into(),
        ])
        .unwrap();
        assert_eq!(f.max_context, Some(8192));
        assert!(f.enable_thinking);
        assert_eq!(f.kv_cache.as_deref(), Some("aura"));
    }

    #[test]
    fn same_id_does_not_bump_loads() {
        let mut id = "model-a".to_string();
        let mut loads = 1usize;
        assert!(swap_ids(&mut id, &mut loads, "model-a", true).is_ok());
        assert_eq!(loads, 1);
        assert_eq!(id, "model-a");
    }

    #[test]
    fn swap_success_drops_previous_id() {
        let mut id = "model-a".to_string();
        let mut loads = 1usize;
        assert!(swap_ids(&mut id, &mut loads, "model-b", true).is_ok());
        assert_eq!(id, "model-b");
        assert_eq!(loads, 2);
    }

    #[test]
    fn failed_load_keeps_previous_id() {
        let mut id = "model-a".to_string();
        let mut loads = 1usize;
        assert!(swap_ids(&mut id, &mut loads, "missing", false).is_err());
        assert_eq!(id, "model-a");
        assert_eq!(loads, 1);
    }

    fn swap_ids(id: &mut String, loads: &mut usize, want: &str, load_ok: bool) -> Result<(), ()> {
        if want == id.as_str() {
            return Ok(());
        }
        if !load_ok {
            return Err(());
        }
        *id = want.to_string();
        *loads += 1;
        Ok(())
    }

    #[test]
    fn serve_flags_max_context_zero_is_err() {
        let err = parse_serve_flags(&["--max-context".into(), "0".into()]).unwrap_err();
        assert!(err.contains("max-context"));
    }

    #[test]
    fn render_no_think_closes_block() {
        // Mutation: ignore enable_thinking and always open-think.
        let msgs = [json!({"role":"user","content":"hi"})];
        let off = render_messages(&msgs, false);
        assert!(off.contains("</think>"));
        let on = render_messages(&msgs, true);
        assert!(!on.contains("</think>"));
    }

    #[test]
    fn event_frame_has_event_name() {
        // Mutation: OpenAI `data:` only : Messages clients never start a block.
        let f = event_frame("message_start", r#"{"type":"message_start"}"#);
        assert!(f.starts_with("event: message_start\r\n"));
        assert!(f.contains("data: {\"type\":\"message_start\"}"));
    }

    #[test]
    fn finish_events_include_content_block_start_and_end_turn() {
        // Mutation: omit content_block_start.
        let ev = messages_finish_events("yo", false, 1, 256);
        assert!(ev.contains("event: content_block_start"));
        assert!(ev.contains("event: content_block_stop"));
        assert!(ev.contains("\"stop_reason\":\"end_turn\""));
        assert!(ev.contains("event: message_stop"));
    }

    #[test]
    fn finish_events_json_leak_is_tool_use() {
        // Mutation: leave markup as text only.
        let leak = r#"<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>"#;
        let ev = messages_finish_events(leak, false, 4, 128);
        assert!(ev.contains("tool_use"));
        assert!(ev.contains("\"stop_reason\":\"tool_use\""));
        assert!(ev.contains("bash"));
    }

    #[test]
    fn flatten_anthropic_system_and_array_user() {
        let v = json!({
            "max_tokens": 32,
            "thinking": {"type": "enabled"},
            "system": "Be terse.",
            "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
        });
        let flat = flatten_anthropic(&v).unwrap();
        assert_eq!(flat[0]["role"], "system");
        assert_eq!(flat[0]["content"], "Be terse.");
        assert_eq!(flat[1]["role"], "user");
        assert_eq!(flat[1]["content"], "hi");
    }

    #[test]
    fn flatten_anthropic_tool_result() {
        let v = json!({
            "max_tokens": 8,
            "messages": [{
                "role":"user",
                "content":[{"type":"tool_result","tool_use_id":"x","content":"ok"}]
            }]
        });
        let flat = flatten_anthropic(&v).unwrap();
        assert_eq!(flat[0]["role"], "tool");
        assert_eq!(flat[0]["content"], "ok");
    }

    #[test]
    fn recover_json_markup_leak() {
        let (name, args, residual) =
            recover_json_markup(r#"pre <tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call> "#)
                .unwrap();
        assert_eq!(name, "bash");
        assert!(args.contains("ls"));
        assert_eq!(residual, "pre");
    }

    #[test]
    fn recover_json_markup_rejects_truncated() {
        assert!(recover_json_markup("<tool_call>{\"name\":\"x\"}").is_none());
    }

    #[test]
    fn think_block_hides_json_leak() {
        let s = "<think><tool_call>{\"name\":\"bash\",\"arguments\":{}}</tool_call></think>ok";
        assert!(recover_tools(s).is_none());
    }

    #[test]
    fn recover_xml_function_leak() {
        let s = "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>";
        let (name, args, _) = recover_tools(s).expect("xml");
        assert_eq!(name, "bash");
        assert!(args.contains("ls"));
    }

    #[test]
    fn watch_disconnect_tcp_drop_sets_abort() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (srv, _) = listener.accept().unwrap();
            let abort = watch_disconnect(&srv);
            let start = std::time::Instant::now();
            while !abort.load(Ordering::Relaxed)
                && start.elapsed() < std::time::Duration::from_secs(2)
            {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let hit = abort.load(Ordering::Relaxed);
            abort.store(true, Ordering::Relaxed);
            hit
        });
        let client = TcpStream::connect(addr).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(40));
        drop(client);
        assert!(t.join().unwrap(), "TCP drop did not set watch_disconnect abort");
    }
}

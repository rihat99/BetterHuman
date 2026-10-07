// Build the docs and render every page to full-page PNGs, light and dark, for self-review.
// Usage: node scripts/docs_screenshots.mjs <out_dir> [--no-build]   ->  <out_dir>/{light,dark}/<page>.png at 1280 px.
// Needs BETTER_ROBOT_CHROME=<chromium / headless-shell binary or wrapper> (same variable as viewer/capture.py).
// Node 22 built-in WebSocket + CDP, no npm deps. The headless shell dlopens libasound.so.2, which this machine
// lacks: wrap the binary in a script that sets LD_LIBRARY_PATH to a directory holding an extracted libasound2 .deb
// (see agent_notes/.../p0_assets/chrome.sh) and point BETTER_ROBOT_CHROME at the wrapper.
import { spawn, spawnSync } from "node:child_process"; import fs from "node:fs"; import path from "node:path"; import http from "node:http";
const args = process.argv.slice(2); const out = path.resolve(args[0]); const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const site = path.join(repo, "site");
if (!process.env.BETTER_ROBOT_CHROME) { console.error("set BETTER_ROBOT_CHROME"); process.exit(2); }
if (!args.includes("--no-build")) { const b = spawnSync("uv", ["run", "mkdocs", "build", "--strict"], { cwd: repo, stdio: "inherit" }); if (b.status !== 0) process.exit(b.status); }
const types = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".png": "image/png", ".webm": "video/webm", ".woff2": "font/woff2" };
const server = http.createServer((q, r) => { let f = path.join(site, decodeURI(q.url.split("?")[0]));
  if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, "index.html");
  fs.readFile(f, (e, d) => { r.writeHead(e ? 404 : 200, { "Content-Type": types[path.extname(f)] ?? "application/octet-stream" }); r.end(d); }); }).listen(0, "127.0.0.1");
await new Promise((ok) => server.on("listening", ok)); const base = `http://127.0.0.1:${server.address().port}`;
const pages = fs.readdirSync(site, { recursive: true }).filter((p) => p.endsWith("index.html") && !p.startsWith("assets")).map((p) => path.dirname(p)).sort();
const chrome = spawn(process.env.BETTER_ROBOT_CHROME, ["--headless", "--no-sandbox", "--remote-debugging-port=0", "--hide-scrollbars", "about:blank"]);
const wsUrl = await new Promise((ok) => chrome.stderr.on("data", (d) => { const m = /ws:\/\/\S+/.exec(d.toString()); if (m) ok(m[0]); }));
const browser = new WebSocket(wsUrl); await new Promise((ok) => (browser.onopen = ok)); let id = 0; const wait = new Map();
browser.onmessage = (e) => { const m = JSON.parse(e.data); if (wait.has(m.id)) { wait.get(m.id)(m.result); wait.delete(m.id); } };
const cdp = (method, params = {}, sessionId) => new Promise((ok) => { wait.set(++id, ok); browser.send(JSON.stringify({ id, method, params, sessionId })); });
const { targetId } = await cdp("Target.createTarget", { url: "about:blank" }); const { sessionId: s } = await cdp("Target.attachToTarget", { targetId, flatten: true });
await cdp("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false }, s);
for (const scheme of ["light", "dark"]) { fs.mkdirSync(path.join(out, scheme), { recursive: true });
  await cdp("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme }] }, s);
  await cdp("Storage.clearDataForOrigin", { origin: base, storageTypes: "all" }, s); // Material caches the palette in localStorage
  for (const p of pages) { await cdp("Page.navigate", { url: `${base}/${p === "." ? "" : p + "/"}` }, s); await new Promise((ok) => setTimeout(ok, 1500)); // CDN MathJax + fonts
    // Material flags the tab bar hidden while its scroll observers settle, which fades the tab labels out in the
    // full-page capture; un-hide it so the shot shows the real tab colours, then let the 0.25 s fade finish.
    await cdp("Runtime.evaluate", { expression: "document.querySelector('.md-tabs')?.removeAttribute('hidden')" }, s);
    await new Promise((ok) => setTimeout(ok, 400));
    const { data } = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true }, s);
    fs.writeFileSync(path.join(out, scheme, (p === "." ? "home" : p.replaceAll("/", "__")) + ".png"), Buffer.from(data, "base64")); } }
console.log(`${pages.length} pages x 2 schemes -> ${out}`); browser.close(); chrome.kill(); server.close();

// Thin proxy plus static hosting. Written fresh rather than ported: AIC's
// server.js carries nine DRES references and pulls in video-metadata.js and
// proxy-target.js, none of which this project needs.
//
//   node server.js          # :3000, proxies /api and /files to :8000

const express = require("express");
const path = require("path");

const app = express();
const PORT = parseInt(process.env.PORT || "3000", 10);
const BACKEND = process.env.API_URL || "http://localhost:8000";

app.use(express.json({ limit: "10mb" }));
app.use(express.static(path.join(__dirname, "public")));

// One generic forwarder for every backend path. Node 20 has global fetch, so
// there is no proxy dependency to install.
async function forward(req, res, backendPath) {
  const url = BACKEND + backendPath;
  try {
    const init = {
      method: req.method,
      headers: { "Content-Type": "application/json" },
    };
    if (req.method !== "GET" && req.method !== "HEAD") {
      init.body = JSON.stringify(req.body ?? {});
    }
    const upstream = await fetch(url, init);
    res.status(upstream.status);

    const type = upstream.headers.get("content-type") || "";
    res.set("content-type", type);
    if (type.startsWith("application/json")) {
      res.send(await upstream.text());
    } else {
      // Keyframes and GIFs come back as bytes.
      res.send(Buffer.from(await upstream.arrayBuffer()));
    }
  } catch (err) {
    res.status(502).json({ error: `backend unreachable at ${url}: ${err.message}` });
  }
}

app.all("/api/*", (req, res) => {
  forward(req, res, req.originalUrl.replace(/^\/api/, ""));
});

// Kept outside /api so <img src> and <video src> can point straight at them.
app.get("/files", (req, res) => forward(req, res, req.originalUrl));
app.get("/gif", (req, res) => forward(req, res, req.originalUrl));

app.listen(PORT, () => {
  console.log(`frontend on http://localhost:${PORT} -> ${BACKEND}`);
});

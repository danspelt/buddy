import "dotenv/config";
import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ---- Active listener arbitration (for later Android tail) ----
let active = null; // "android" | "windows" | null
const wss = new WebSocketServer({ port: 8081 });
wss.on("connection", (ws) => {
  ws.on("message", (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      if (msg.type === "claim") {
        if (!active) {
          active = msg.device;
          ws.send(JSON.stringify({ type: "granted" }));
        } else {
          ws.send(JSON.stringify({ type: "denied", active }));
        }
      }
      if (msg.type === "release" && active === msg.device) {
        active = null;
      }
    } catch (e) {
      console.error("WS message error", e);
    }
  });
});

// ---- Chat endpoint ----
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.post("/chat", async (req, res) => {
  try {
    const { text, device } = req.body || {};
    const sys =
      process.env.SYSTEM_PROMPT ||
      "You are Brighton, a calm, supportive voice companion. Keep replies brief and kind.";

    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || "gpt-4o-mini",
      messages: [
        { role: "system", content: sys },
        { role: "user", content: text || "" },
      ],
    });

    const reply_text = completion.choices?.[0]?.message?.content || "I'm here.";
    res.json({ reply_text });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

const port = process.env.PORT || 8080;
app.listen(port, () =>
  console.log(`Buddy brain on http://localhost:${port}  (WS :8081)`)
);

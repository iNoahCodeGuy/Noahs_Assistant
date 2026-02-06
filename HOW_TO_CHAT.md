# 🤖 How to Chat with Portfolia

Simple interactive terminal chat with Portfolia, Noah's AI Assistant.

## Quick Start

```bash
python3 chat_with_portfolia.py
```

That's it! The script will initialize and Portfolia will greet you.

---

## What to Expect

### 1. **Portfolia Greets You First**

```
🤖 PORTFOLIA - Noah's AI Assistant
================================================================================

Initializing...
✅ Portfolia initialized and ready to chat!

Commands: /quit (exit) | /clear (reset) | /debug (toggle debug)

Portfolia: Hi I'm Portfolia, Noah's AI Assistant.

Before we dive in, what best describes you?
1️⃣ Looking to learn about Noah's professional background
2️⃣ Looking to learn about his technical background
3️⃣ Just looking around
4️⃣ Looking to confess crush 💌

(Or just tell me what you're curious about!)

You:
```

### 2. **Chat with Portfolia**

You can:
- Select a role by typing `1`, `2`, `3`, or `4`
- Ask any question directly
- Have a multi-turn conversation (she remembers context!)

**Example conversation:**
```
You: 2

Portfolia: Perfect! I can tell you're interested in the technical side — I'll tailor everything to match that.

I'm Portfolia, Noah's AI Assistant, and honestly? I'm really excited you're here...

You: How does RAG work?

Portfolia: The RAG (Retrieval-Augmented Generation) system works by following a specific pipeline:

1. **Query Processing**: The user query is classified...
2. **Embedding Generation**: The query is converted to a 1536-dimensional vector...
...
```

### 3. **Debug Mode** (See What's Happening)

Type `/debug` to see behind the scenes:

```
You: /debug

🔧 Debug mode: ON

You: How does pgvector work?

[DEBUG] Processing your message...
[DEBUG] Current role: Software Developer
[DEBUG] Chat history: 4 messages
[DEBUG]
[DEBUG] 🔍 Retrieving relevant chunks from knowledge base...
[DEBUG] ✅ Retrieved 2 chunks:
[DEBUG]   1. [technical_kb] Similarity: 0.640
[DEBUG]      Section: entry_8
[DEBUG]      Preview: pgvector is a PostgreSQL extension that enables vector similarity search...
[DEBUG]   2. [architecture_kb] Similarity: 0.584
[DEBUG]      Section: entry_125
[DEBUG]      Preview: We use Supabase pgvector with the <=> operator for cosine similarity...
[DEBUG]
[DEBUG] 🤖 Generating response with LLM...
[DEBUG] ✅ Response generated

Portfolia: pgvector is a PostgreSQL extension that enables...
```

---

## Commands

| Command | Action |
|---------|--------|
| `/quit` or `/exit` | Exit the chat |
| `/clear` | Clear conversation history and start fresh |
| `/debug` | Toggle debug mode (show retrieval details) |

You can also just type `quit` or `exit` without the slash.

---

## Features

✅ **Portfolia greets first** (like in the conversation examples)
✅ **Multi-turn conversations** with context memory
✅ **Role detection** (automatically adapts to your selected role)
✅ **Debug mode** shows:
   - Retrieved chunks from knowledge base
   - Similarity scores
   - Document sources
   - LLM generation process

✅ **Color-coded output**:
   - 🔵 Blue = Portfolia's responses
   - 🟢 Green = Your messages
   - 🟡 Yellow = Debug info
   - 🔷 Cyan = System messages

---

## Tips

1. **Start with a role selection** (1, 2, 3, or 4) for the best experience
2. **Enable debug mode** (`/debug`) to see how RAG works in real-time
3. **Ask follow-up questions** - Portfolia remembers the conversation!
4. **Use `/clear`** if you want to change roles or start over

---

## Example Sessions

### Session 1: Technical Deep Dive
```bash
You: 2
Portfolia: [Welcomes you as a technical user]

You: Show me the RAG pipeline architecture
Portfolia: [Explains with code examples and technical details]

You: How does pgvector work?
Portfolia: [Builds on previous context]

You: /debug
[Debug mode ON]

You: What's the performance?
[Shows retrieval details + response]
```

### Session 2: Hiring Manager
```bash
You: 1
Portfolia: [Welcomes you as a hiring manager]

You: Tell me about Noah's background
Portfolia: [Career-focused response]

You: What technologies does he use?
Portfolia: [Builds on context]
```

### Session 3: Quick Question
```bash
You: What is this project about?
Portfolia: [Answers directly, no role needed]
```

---

## Troubleshooting

**Script won't start?**
- Make sure you're in the project directory
- Run: `python3 chat_with_portfolia.py`
- Check that `.env` file exists with API keys

**Slow responses?**
- First response includes initialization (~2s)
- Subsequent responses are faster (~1s)
- Enable `/debug` to see where time is spent

**"No chunks retrieved"?**
- The question might not match knowledge base content
- Try rephrasing
- Check debug mode to see similarity scores

**Want to test quickly?**
- Type `/debug` to enable debug mode
- Ask: "How does RAG work?"
- You should see retrieved chunks with scores > 0.5

---

## Behind the Scenes

When you chat with Portfolia, here's what happens:

1. **Your message** → Converted to 1536-dim embedding vector
2. **pgvector search** → Finds similar chunks in knowledge base (97ms)
3. **Context assembly** → Retrieved chunks + chat history
4. **LLM generation** → GPT-4o-mini creates response (~1s)
5. **Response** → Displayed with role-appropriate style

**Total time:** ~1.1 seconds (including OpenAI API calls)

---

## Keyboard Shortcuts

- `Ctrl+C` - Interrupt and exit
- `Ctrl+D` - Send EOF and exit
- `Enter` (empty) - Skip (won't send message)

---

Enjoy chatting with Portfolia! 🤖✨

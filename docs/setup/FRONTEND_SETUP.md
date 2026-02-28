# 🎨 Next.js Frontend Setup

## ✅ What Was Created

A professional ChatGPT/Grok-style frontend with:

- ✨ **Modern Dark Theme** - Purple/pink gradient background
- 💬 **Chat Interface** - Message bubbles, streaming responses, loading states
- 🎭 **Role Selector** - Dropdown to switch between user types
- 📚 **Source Citations** - Shows knowledge base sources with similarity scores
- 📱 **Responsive Design** - Works on mobile and desktop
- ⚡ **Vercel-Ready** - Optimized for deployment

## 🚀 Quick Start (Local Development)

### Prerequisites
1. **Install Node.js**: https://nodejs.org/ (v18 or higher)
2. Verify installation:
   ```powershell
   node --version  # Should show v18.x.x or higher
   npm --version   # Should show 9.x.x or higher
   ```

### Setup Steps

```powershell
# 1. Install frontend dependencies
npm install

# 2. Run development server
npm run dev

# 3. Open browser
# Visit: http://localhost:3000
```

## 🔗 Connecting to Python Backend

The frontend calls `/api/chat` which currently returns mock data. To connect to your Python backend:

### Option 1: Update API Route (Recommended)

Edit `app/api/chat/route.ts` and uncomment the real backend call:

```typescript
// Uncomment this section:
const response = await fetch(pythonBackendUrl, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body)
})
```

Then set environment variable:
```powershell
# Create .env.local file
echo "PYTHON_BACKEND_URL=http://localhost:8000" > .env.local
```

### Option 2: Deploy Python Backend Separately

1. Deploy Python backend to Railway/Render/Fly.io
2. Get the deployment URL (e.g., `https://your-backend.railway.app`)
3. Update `.env.local`:
   ```
   PYTHON_BACKEND_URL=https://your-backend.railway.app
   ```

## 📦 Project Structure

```
app/
├── globals.css          # Tailwind + custom styles
├── layout.tsx           # Root layout with metadata
├── page.tsx             # Main chat interface (ChatGPT-style)
└── api/
    └── chat/
        └── route.ts     # API endpoint for chat

package.json             # Dependencies
tsconfig.json            # TypeScript config
tailwind.config.js       # Tailwind theme (dark mode, gradients)
next.config.js           # Next.js configuration
vercel.json              # Vercel deployment config
```

## 🎨 Design Features

### Colors
- **Background**: Deep black (`#0A0A0A`)
- **Surface**: Dark gray (`#1A1A1A`)
- **Primary**: Purple (`#8B5CF6`)
- **Secondary**: Pink (`#EC4899`)
- **Gradients**: Purple → Pink throughout

### Components
- **Message Bubbles**: User (gradient) vs Assistant (dark surface)
- **Avatars**: Gradient circles with icons
- **Loading**: Bouncing dots animation
- **Sources**: Collapsible citation panel

## 🚀 Deployment to Vercel

### Automatic (Git Push)
```powershell
git add .
git commit -m "feat: Add Next.js frontend"
git push origin data_collection_management
```

Vercel will automatically:
1. Detect Next.js app
2. Install dependencies
3. Build and deploy
4. Give you a live URL

### Manual (Vercel Dashboard)
1. Go to https://vercel.com/dashboard
2. Import your GitHub repo
3. Vercel auto-detects Next.js
4. Click "Deploy"

### Environment Variables in Vercel
In Vercel Dashboard → Settings → Environment Variables:
```
PYTHON_BACKEND_URL = https://your-python-backend-url.com
```

## 🔧 Troubleshooting

### "npm not found"
**Install Node.js**: https://nodejs.org/

### Build errors
```powershell
# Clear cache and reinstall
rm -r node_modules
rm package-lock.json
npm install
```

### TypeScript errors in IDE
```powershell
# Install types
npm install --save-dev @types/node @types/react @types/react-dom
```

## 📝 Next Steps

1. ✅ Install Node.js
2. ✅ Run `npm install`
3. ✅ Test locally with `npm run dev`
4. ✅ Connect to Python backend
5. ✅ Push to GitHub
6. ✅ Vercel auto-deploys!

## 🎯 Features to Add Later

- [ ] Streaming responses (SSE)
- [ ] Voice input/output
- [ ] Chat history persistence
- [ ] Export conversation
- [ ] Dark/light theme toggle
- [ ] Mobile app wrapper

---

**Ready to deploy!** Once you install Node.js and run `npm install`, everything is set up! 🎉

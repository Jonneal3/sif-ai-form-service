# Vercel Environment Variables

Copy these into Vercel Dashboard → Project → Settings → Environment Variables

## Required

### Supabase
```
SUPABASE_URL=https://xvpagpzufitqzoijoalz.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh2cGFncHp1Zml0cXpvaWpvYWx6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzQyMzg1MCwiZXhwIjoyMDYyOTk5ODUwfQ.Uzh0xJN5GP5koSSHtbtr1kDP8q1AKQBZk34G_-Vj8j0
```

### DSPy (LLM Provider)
```
DSPY_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

## Optional

```
DSPY_MODEL_LOCK=llama-3.3-70b-versatile
DSPY_TEMPERATURE=0.7
DSPY_LLM_TIMEOUT_SEC=20
DSPY_NEXT_STEPS_MAX_TOKENS=2000
```

---

## Quick Copy-Paste (one at a time in Vercel UI)

**SUPABASE_URL**
```
https://xvpagpzufitqzoijoalz.supabase.co
```

**SUPABASE_SERVICE_ROLE_KEY**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh2cGFncHp1Zml0cXpvaWpvYWx6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzQyMzg1MCwiZXhwIjoyMDYyOTk5ODUwfQ.Uzh0xJN5GP5koSSHtbtr1kDP8q1AKQBZk34G_-Vj8j0
```

**DSPY_PROVIDER**
```
groq
```

**GROQ_API_KEY**
```
(paste your Groq API key here)
```


# 🚀 GUÍA MAESTRA DE INICIO (THE BIBLE)

Para que el sistema genere videos al 100%, necesitas tener **5 TERMINALES** abiertas.

## 1. Infraestructura (Terminal 1)
**Qué hace:** Enciende la Base de Datos y la Cola de Trabajos.
```powershell
docker start tohjo-redis supabase_db_videos
```
✅ **Éxito si:** Docker Desktop muestra los contenedores en verde.

---

## 2. API "Cerebro" (Terminal 2)
**Qué hace:** Gestiona todo el sistema.
⚠️ **IMPORTANTE:** Usa `pnpm` aquí.
```powershell
cd apps/api
pnpm install
pnpm run dev
```
✅ **Éxito si ves:** `Server listening at http://0.0.0.0:3001`

---

## 3. Web "Interfaz" (Terminal 3)
**Qué hace:** La página web donde subes los archivos.
⚠️ **IMPORTANTE:** Usa `pnpm` aquí.
```powershell
cd apps/web
pnpm install
pnpm run dev
```
✅ **Éxito si ves:** `Ready in ... ms`

---

## 4. Worker Audio "Oídos" (Terminal 4)
**Qué hace:** Escucha, separa voces y alinea texto (Python).
```powershell
cd workers/audio-cpu
.\.venv\Scripts\activate
python main.py
```
✅ **Éxito si ves:** `Worker initialized. Entering loop...`

---

## 5. Worker Render "Ojos" (Terminal 5)
**Qué hace:** Crea el video final (Remotion).
```powershell
cd workers/render
pnpm run start
```
✅ **Éxito si ves:** `Render worker listening on Redis`

---

## 💡 Resumen Rápido
| Componente | Carpeta | Comando |
| :--- | :--- | :--- |
| **Bases de Datos** | (Cualquiera) | `docker start ...` |
| **API** | `apps/api` | `pnpm run dev` |
| **Web** | `apps/web` | `pnpm run dev` |
| **Audio (Python)** | `workers/audio-cpu` | `python main.py` |
| **Render (Video)** | `workers/render` | `pnpm run start` |

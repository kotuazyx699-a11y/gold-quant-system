// =========================================================================================
// Gold Quant V14 — Service Worker (PWA โหมดแอปมือถือเต็มจอ)
// กลยุทธ์: Network-First ทุกเส้นทาง เพื่อให้หน้าเว็บได้ของสดใหม่เสมอ (สำรอง cache เฉพาะตอนออฟไลน์)
// ยกเว้นเด็ดขาด (ห้ามแคช 100%): data/state.json, api.github.com, gist.githubusercontent.com
//   → ราคา Spot / สถานะออเดอร์ / Canonical Snapshot ต้องสดจริงเสมอ ไม่ถูกหน่วงด้วย cache
// หมายเหตุ: bump CACHE_NAME ทุกครั้งที่แก้ไฟล์หน้าเว็บครั้งใหญ่ เพื่อล้าง cache เก่า
// =========================================================================================

const CACHE_NAME = 'gold-quant-v14-2026-11-24';

// พรีแคชเฉพาะเปลือกแอป (offline shell) — ไม่รวมข้อมูลสดใด ๆ
const PRECACHE_URLS = ['./index.html', './manifest.json', './icon.svg'];

// โดเมนที่ห้ามแคชเด็ดขาด (ข้อมูลสดตลอดเวลา)
const NEVER_CACHE_HOSTS = [
  'api.github.com',
  'gist.githubusercontent.com',
  'gist.github.com',
  'ntfy.sh'
];

// พาธที่ห้ามแคชเด็ดขาด
const NEVER_CACHE_PATH_SUFFIXES = [
  '/data/state.json',
  'state.json',
  '/data/gold_m15_vault.csv'
];

function mustBypass(urlStr) {
  try {
    const u = new URL(urlStr);
    if (NEVER_CACHE_HOSTS.some(h => u.hostname === h || u.hostname.endsWith('.' + h))) return true;
    if (NEVER_CACHE_PATH_SUFFIXES.some(p => u.pathname.endsWith(p))) return true;
    return false;
  } catch (e) {
    return false;
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE_URLS).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // ไม่แตะคำขอที่ไม่ใช่ GET (เช่น PATCH/POST ซิงก์ Gist, ntfy) — วิ่งตรงเสมอ
  if (req.method !== 'GET') return;

  let url;
  try { url = new URL(req.url); } catch (e) { return; }

  // ปล่อย ws/wss/data:/etc. ให้เบราว์เซอร์จัดการเอง (WebSocket ราคาสดห้ามถูกแตะ)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  // state.json / GitHub API / Gist raw → Network only 100% (ไม่แคช ไม่ fallback cache)
  if (mustBypass(req.url)) return;

  // Network-First: ยิงเน็ตก่อนเสมอ แล้วเก็บสำรองไว้ใช้ตอนออฟไลน์เท่านั้น
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok && res.type === 'basic') {
          const copy = res.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(async () => {
        const cached = await caches.match(req);
        if (cached) return cached;
        if (req.mode === 'navigate') {
          const shell = await caches.match('./index.html');
          if (shell) return shell;
        }
        return Response.error();
      })
  );
});

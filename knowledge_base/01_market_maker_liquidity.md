# Market Maker & Liquidity Sweep Rules (Jane Street / Institutional Framework)
- สภาพคล่อง (Liquidity Pools) มักสะสมอยู่เหนือ Previous Day High (PDH) และใต้ Previous Day Low (PDL) รวมถึงขอบ Value Area (VAH/VAL)
- Liquidity Sweep Setup: เมื่อราคาพุ่งทะลุ PDH หรือ PDL เพื่อกวาด Stop Loss ของรายย่อย แต่แท่งเทียน M15 ปิดกลับเข้ามาในกรอบเดิมพร้อมแท่งยืนยันกลับตัว ถือเป็นสัญญาณกลับตัวที่มี R:R สูง
- ห้ามไล่ราคา (Chasing) กลางอากาศ ต้องรอเข้าเฉพาะบริเวณจุดอ้างอิงที่มีโครงสร้างรองรับ (PDH, PDL, EMA50, VAH, VAL) และอยู่ในช่วง London/NY Killzone (13:00 - 00:30 น. เวลาไทย)

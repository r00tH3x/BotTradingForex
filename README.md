# Bot Trading Forex Pro
Versi: v2.0  
Penulis / Developer: _ibar_  
Tanggal: _21/10/2025_

## Deskripsi
Bot ini adalah **Telegram Bot** yang dirancang untuk membantu analisa dan eksekusi sinyal trading forex dengan fitur otomatisasi dan analisis teknikal.

### Kegunaan Utama
- Menyediakan **live signals**
- **Market scanner** otomatis
- **Watchlist monitoring**
- **Analisis Fibonacci**
- **Integrasi kalender ekonomi**
- **Manajemen risiko dan edukasi trading**

## Fitur Utama
- Analisis multi-timeframe (M5 - D1)
- Deteksi konfluensi Fibonacci
- Deteksi Break of Structure (BOS)
- Konfirmasi volume
- Integrasi kalender ekonomi
- Command Telegram interaktif: `/start`, `/signals`, `/scan`, `/watchlist`, `/fib`, `/help`

## Persyaratan Sistem
- Python 3.8+
- Token bot Telegram dari BotFather
- Akses internet
- (Opsional) API data pasar / kalender ekonomi

## Instalasi
1. Clone repository
   ```bash
   git clone https://github.com/r00tH3x/BotTradingForex.git
   cd BotTradingForex
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Ganti token bot Telegram
   ```python
   BOT_TOKEN = "ISI_TOKEN_BOT_TELEGRAM_KAMU"
   ALPHA_VANTAGE_KEY = "your_api_key"
   FOREX_API_KEY = "your_api_key"
   NEWS_API_KEY = "your_api_key"
   ```
4. Jalankan bot
   ```bash
   python app.py
   ```

## Cara Penggunaan
- `/start` → menampilkan menu utama
- `/signals` → menampilkan sinyal trading
- `/scan` → memindai pasar forex
- `/watchlist` → menampilkan watchlist
- `/fib PAIR` → analisis Fibonacci pada pair tertentu
- `/help` → panduan lengkap

## Struktur Kode
- `app.py` : file utama

## Manajemen Risiko
Gunakan bot ini hanya sebagai alat bantu analisis. Tidak menjamin profit. Risiko trading tetap ditanggung pengguna.

## Contribusi
1. Fork repository
2. Buat branch baru: `git checkout -b feature-namaFitur`
3. Commit dan push
4. Buat Pull Request

## Lisensi
MIT License  
© 2025 ibar

## Kontak
- GitHub: https://github.com/r00tH3x/BotTradingForex
- Instagram: @Ibar____

---
Gunakan bot ini dengan bijak dan tetap disiplin dalam manajemen risiko.

@echo off
echo ============================================
echo  Building GGUF Benchmark Tool v2.4.1
echo ============================================
echo.

pip install pyinstaller

echo.
echo Generating application icon...
python make_icon.py

echo.
echo Building standalone EXE...
python -m PyInstaller --onefile --windowed --name "GGUFBenchmark_v2.4.1" ^
  --icon "icon.ico" ^
  --add-data "icon.ico;." ^
  --hidden-import gguf_reader ^
  --hidden-import system_info ^
  gguf_benchmark.py

echo.
echo ============================================
echo  Build complete!
echo  EXE location: dist\GGUFBenchmark_v2.4.1.exe
echo ============================================
pause

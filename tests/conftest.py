import os
import sys

# Permite rodar os testes do repo sem instalar (acha o pacote em ../src).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

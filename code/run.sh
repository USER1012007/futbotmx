#!/bin/bash

python main.py > salida.log 2>&1
python test/mock_tracking_visualization.py > salida_tracking.log 2>&1

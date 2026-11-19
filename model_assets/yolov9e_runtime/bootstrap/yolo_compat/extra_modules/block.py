"""Resolve legacy YOLOv9 layer names to built-in Ultralytics 8.2.50 layers.

This is a compatibility experiment, not the paper authors' custom source.
"""
from ultralytics.nn.modules.block import ADown, CBLinear, CBFuse, RepNCSPELAN4
from ultralytics.nn.modules.block import RepCSP as RepNCSP
from ultralytics.nn.modules.block import RepBottleneck as RepNBottleneck
from ultralytics.nn.modules.conv import RepConv as RepConvN

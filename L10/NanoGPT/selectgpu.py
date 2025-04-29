#!/usr/bin/env python3

import sys
import numpy as np

# CEF: added device selector and check for avail GPU mem...
def SelectGPU():
	
	def Print(msg):
		print(msg, file=sys.stderr)
	
	def ToMiB(sz):
		s = int(1.0*sz/1024**2+.5)
		return sz, str(s) + " MiB"

	def GetGPUMem(device, debug=False):
		with torch.cuda.device(device):
			info = torch.cuda.mem_get_info()
			total = ToMiB(info[1])
			free  = ToMiB(info[0])
			allocated = ToMiB(info[1]-info[0])
			if debug:
				Print(f"DEVICE[{device}]: total mem={total},  free={free}, allocated={allocated}")
			return free

	gpus = 0
	try:
		import torch
		gpus = torch.cuda.device_count()
	except ImportError as ex:
		Print("WARNING: could not import 'torch'..")
		gpus = -1
						
	if gpus <= 0:
		Print("WARNING: no GPU present, selecting CPU instead so expect slow training..")
		return "cpu"
			
	gpumem = [GetGPUMem(i)[0] for i in range(gpus)]

	selected_device = np.argmax(gpumem)
	selected_device_freemem = gpumem[selected_device]
	selected_device_freemem_threshold = int(1.5*1024**3) # need at least 1.5 GiB to begin training

	if selected_device_freemem < selected_device_freemem_threshold:
		Print(f"ERROR: will not begin training due to lack of GPU memory, needs at least {ToMiB(selected_device_freemem)[1]}..")
		sys.exit(-1)

	selected_device = "cuda:" + str(selected_device)
	Print(f"SELECTED device '{selected_device}' with free mem {ToMiB(selected_device_freemem)[1]} out of total {gpus} GPUs")
	return selected_device

def TestSelectGPU():
	d = SelectGPU()
	print(f"TEST: found device '{d}'")
	assert isinstance(d, str)
	assert len(d) > 0
	assert d=="cpu" or d.find("cuda")==0

if __name__=='__main__':
    try:
        TestSelectGPU()
    except Exception as ex:
        print(f"ERROR: exception occured, ex={ex}", file=sys.stderr)
        sys.exit(-1)

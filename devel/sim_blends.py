import os.path
import numpy as np
import galsim
import yaml
import footprints
import jiffy


class Arguments:
	"""docstring for Arguments"""
	def __init__(self, config_file, footprint_number):
		self.config_file = config_file
		self.footprint_number = footprint_number
		

def main():
	config_file = "../config/jiffy_blend.yaml"
	config = yaml.load(open(config_file))
	rstr = jiffy.Roaster(config)
	rstr.initialize_param_values(config["init"]["init_param_file"])

	img = rstr.make_data()
	galsim.fits.write(img, os.path.splitext(config["io"]["infile"])[0] + ".fits")

	args = Arguments(config_file, 0)
	# jiffy.do_roaster_sampling(args, rstr)


if __name__ == '__main__':
	main()

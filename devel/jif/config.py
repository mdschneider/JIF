"""
Sub-class the Python ConfigParser so we can specify defaults on read
"""
from ConfigParser import RawConfigParser, NoOptionError

# http://stackoverflow.com/a/35807133
class DefConfigParser(RawConfigParser):
    def get(self, section, option, default=None):
        try:
            return RawConfigParser.get(self, section, option)
        except NoOptionError:
            return default
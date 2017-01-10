import merge_streams
from datetime import datetime
from collections import OrderedDict
import json


class Cat:
    def __init__(self, name):
        self.mappings = {}
        self.newID = 0
        self.name = name
        self.type = 'categorical'

    @property
    def num_classes(self):
        return len(self.mappings)

    def __call__(self, feature):
        if feature not in self.mappings:
            self.mappings[feature] = self.newID
            self.newID += 1
        return [self.mappings[feature]]


class Time:
    def __init__(self, date_format):
        self.date_format = date_format
        self.name = 'time'
        self.type = 'meta'

    def __call__(self, feature):
        date = datetime.strptime(feature, self.date_format)
        return ["%s%s%s" % (date.year, date.month, date.day)]


class Weekday:

    def __init__(self, date_format):
        self.name = 'weekday'
        self.date_format = date_format
        self.type = 'categorical'
        self.num_classes = 7

    def __call__(self, x):
        return [datetime.strptime(x, self.date_format).weekday()]


class Ident:

    def __init__(self, name):
        self.name = name
        self.type = 'continuous'

    def __call__(self, feature):
        return [feature]

data = merge_streams.Merge(filepath='./', file_list=['crisp_proto.csv'],
                           sort_column='period', date_format='int', delimiter=',')
header = data.headers[0]
idxs = {column: idx for idx, column in enumerate(header)}
mappings = OrderedDict([('period', Time('%Y%m%d')),
                        ('day', Weekday('%Y-%m-%d')),
                        ('saddr', Cat('saddr')),
                        ('daddr', Cat('daddr')),
                        ('dport', Cat('dport')),
                        ('proto', Cat('proto')),
                        ('category', Cat('category')),
                        ('deployment', Cat('deployment')),
                        ('times_seen', Ident('times_seen'))])

output_header = []
for m in mappings.values():
    output_header.append(m.name)
print(output_header)

for event_type, event in data():
    vector = []
    for column, mapping in mappings.iteritems():
        vector += mapping(event[idxs[column]])
    print(vector)

feature_specs = {'num_features': len(output_header),
                 'continuous': []}
for idx, m in enumerate(mappings.values()):
    if m.type == 'categorical':
        feature_specs[m.name] = {}
        feature_specs[m.name]['num_classes'] = m.num_classes
        feature_specs[m.name]['index'] = idx
    if m.type == 'meta':
        feature_specs[m.name] = [idx]
    if m.type == 'continuous':
        feature_specs['continuous'].append(idx)

with open('crisp_feature_spec', 'w') as f:
    json.dump(feature_specs, f)

#   {"name": "day",   "transform": "time"},
#   {"name": "period", "transform": "DaysFromStart"},
#   {"name": "saddr", "transform": "CatOOV"},
#   {"name": "daddr", "transform": "CatOOV"},
#   {"name": "dport", "transform": "CatOOV"},
#   {"name": "proto", "transform": "CatOOV"},
#   {"name": "category", "transform": "CatOOV"},
#   {"name": "deployment", "transform": "CatOOV"},
#   {"name": "site", "transform": "CatOOV"},
#   {"name": "times_seen", "transform": "ident"},
#   {"name": "first_seen_dtm", "transform": "DaysFrom1971"},
#   {"name": "last_seen_dtm", "transform": "DaysFrom1971"},
#   {"name": "times_alerted", "transform": "ident"},
#   {"name": "first_alerted_dtm", "transform": "DaysFrom1971"},
#   {"name": "last_alerted_dtm", "transform": "DaysFrom1971"},
#   {"name": "sum_spayload", "transform": "ident"},
#   {"name": "sum_dpayload", "transform": "ident"},
#   {"name": "sum_sbytes", "transform": "ident"},
#   {"name": "sum_dbytes", "transform": "ident"},
#   {"name": "sum_spackets", "transform": "ident"},
#   {"name": "sum_dpackets", "transform": "ident"},
#   {"name": "sum_duration", "transform": "ident"},
#   {"name": "min_duration", "transform": "ident"},
#   {"name": "max_duration", "transform": "ident"},
#   {"name": "scountrycode", "transform": "CatOOV"},
#   {"name": "sorganization", "transform": "CatOOV"},
#   {"name": "slat", "transform": "ident"},
#   {"name": "slong", "transform": "ident"},
#   {"name": "dcountrycode", "transform": "CatOOV"},
#   {"name": "dorganization", "transform": "CatOOV"},
#   {"name": "dlat", "transform": "ident"},
#   {"name": "dlong", "transform": "ident"},
#   {"name": "distance", "transform": "ident"},
#   {"name": "src_internal", "transform": "CatOOV"},
#   {"name": "dst_internal", "transform": "CatOOV"},
#   {"name": "bucket", "transform": "CatOOV"},
#   {"name": "virtual_site", "transform": "CatOOV"},
#   {"name": "covmap_src_deployment", "transform": "CatOOV"},
#   {"name": "covmap_dst_deployment", "transform": "CatOOV"}

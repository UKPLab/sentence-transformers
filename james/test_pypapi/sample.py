from pypapi import papi_low as papi
from pypapi import events

papi.library_init()

evs = papi.create_eventset()
papi.add_event(evs, events.PAPI_FP_OPS)

papi.start(evs)

q = 0
for i in range(2000):
  q += i / 23

result = papi.stop(evs)
print(result)

papi.cleanup_eventset(evs)
papi.destroy_eventset(evs)

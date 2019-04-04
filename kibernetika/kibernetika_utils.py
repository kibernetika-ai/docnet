import os
import sys


class _FakeKibernetikaMgr(object):
    def __init__(self):
        pass

    def update_task_info(self, submit):
        pass

    def build_id(self):
        return None


class KibernetikaMgr(object):
    def __init__(self):
        if os.environ.get('PROJECT_ID', None) is not None:
            from mlboardclient.api import client
            self._cl = client
        else:
            self._cl = _FakeKibernetikaMgr()

    def build_id(self):
        return os.environ.get('BUILD_ID', None)

    def update_task_info(self, submit):
        try:
            self._cl.update_task_info(submit)
        except:
            print('Unexpected error during submit state: {}'.format(sys.exc_info()[0]))


klclient = KibernetikaMgr()

import unittest
from tools.analysis.profile_summary import summarize

def fixture():
    rows=[('initialization',2,0),('fit',3,2),('loop',5,1),('tracking',4,1),('slam',10,0),('evaluation',6,0)]
    return dict(status='completed',profile='stages',timings=[dict(stage=s,seconds=t,nesting_depth=d,status='completed') for s,t,d in rows])

class ProfileTests(unittest.TestCase):
    def test_exclusive_partition(self):
        r=summarize(fixture())
        self.assertEqual(r['exclusive_partition_seconds'],10)
        self.assertEqual(r['stages']['loop']['exclusive_seconds'],2)
        self.assertEqual(r['stages']['slam']['exclusive_seconds'],1)
        self.assertNotIn('evaluation',r['stages'])
    def test_missing_parent(self):
        r=fixture();r['timings']=[row for row in r['timings'] if row['nesting_depth']!=1]
        with self.assertRaises(ValueError):summarize(r)
    def test_impossible_child_duration(self):
        r=fixture();r['timings'][1]['seconds']=7
        with self.assertRaises(ValueError):summarize(r)
    def test_wall_mode_rejected(self):
        r=fixture();r['profile']='wall'
        with self.assertRaises(ValueError):summarize(r)

if __name__=='__main__':unittest.main()

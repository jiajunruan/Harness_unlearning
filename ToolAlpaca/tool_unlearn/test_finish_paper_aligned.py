"""Offline continuation coverage and integrity checks; no models or API calls."""
from copy import deepcopy
import unittest

from tool_unlearn.finish_paper_aligned import completed_groups, split_remaining, validate_traces


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.source = [dict(Name=name, Instructions=[f'{name}{i}' for i in range(n)],
                            Golden_Answers=[[{'Action': name, 'Action_Input': '{}'}] for _ in range(n)],
                            NLDocumentation='documentation', Documentation='spec')
                       for name, n in [('A', 2), ('B', 3), ('C', 2), ('D', 1)]]

    def test_disjoint_complete_split_preserves_source(self):
        original = deepcopy(self.source)
        partial = deepcopy(self.source)
        partial[0]['Instances'] = [{}, {}]
        done = completed_groups(self.source, partial)
        shards = split_remaining(self.source, done)
        names = [set(done)] + [{r['Name'] for r in shard} for shard in shards]
        self.assertEqual(set.union(*names), {'A', 'B', 'C', 'D'})
        for i in range(3):
            for j in range(i):
                self.assertFalse(names[i] & names[j])
        self.assertEqual(self.source, original)
        self.assertTrue(all('Instances' not in row for shard in shards for row in shard))
        merged = [done.get(row['Name'], row) for row in self.source]
        self.assertEqual([r['Name'] for r in merged], ['A', 'B', 'C', 'D'])

    def test_metadata_change_and_partial_group_rejected(self):
        partial = deepcopy(self.source)
        partial[0]['Instructions'][0] = 'changed'
        with self.assertRaises(ValueError):
            completed_groups(self.source, partial)
        partial = deepcopy(self.source)
        partial[0]['Instances'] = [{}]
        with self.assertRaises(ValueError):
            completed_groups(self.source, partial)

    def test_trace_gold_duplicate_and_coverage_checks(self):
        traces = [dict(api=row['Name'], instruction_id=i, instruction=instruction,
                       golden_answers=row['Golden_Answers'][i])
                  for row in self.source for i, instruction in enumerate(row['Instructions'])]
        self.assertEqual(validate_traces(self.source, list(reversed(traces))), traces)
        for invalid in (traces[:-1], traces + [traces[0]]):
            with self.assertRaises(ValueError):
                validate_traces(self.source, invalid)
        invalid = deepcopy(traces)
        invalid[0]['golden_answers'] = []
        with self.assertRaises(ValueError):
            validate_traces(self.source, invalid)


if __name__ == '__main__':
    unittest.main()

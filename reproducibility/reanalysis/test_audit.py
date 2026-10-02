"""Regression tests for the revised estimands, pairing, and resampling units."""
import unittest
import numpy as np
import audit


def row(condition, initial='Yes', final='Yes', truth='Yes', **extra):
    r = dict(condition=condition, question_id='q1', duplicate=False, valid=True,
             pass1=initial, pass2=final, ground_truth=truth, input_sha256='fixed',
             opponent_answer=None, pass1_record_id=None, replicate_id=None,
             served_model=None, provider=None, request_normalized_sha256=None,
             delivered_input_verified=False, delivered_rationale_verified=False,
             document_id='doc1', domain='Medicine')
    r.update(extra)
    return r


class AuditTests(unittest.TestCase):
    def test_shared_initial_required_for_every_intersection(self):
        rows = [row('A0'), row('A1'), row('A3', initial='No')]
        for conditions in [['A1','A3'],['A0','A1','A3']]:
            pairs, excluded, _ = audit.pair_rows(rows,conditions)
            self.assertEqual(pairs,[])
            self.assertEqual(excluded[0]['exclusion_reason'],'mismatched_initial')

    def test_invalid_first_attempt_not_replaced_by_valid_retry(self):
        rows = [row('A1', valid=False),row('A1',duplicate=True),row('A3')]
        pairs, excluded, _ = audit.pair_rows(rows,['A1','A3'])
        self.assertEqual(pairs,[])
        self.assertEqual(excluded[0]['exclusion_reason'],'invalid_branch_output')

    def test_wrong_opponent_rejected(self):
        pairs, excluded, _ = audit.pair_rows([row('A1',opponent_answer='No'),
                                             row('A3',opponent_answer='Yes')],['A1','A3'])
        self.assertFalse(pairs)
        self.assertEqual(excluded[0]['exclusion_reason'],'mismatched_opponent')

    def test_unknown_identity_is_not_a_made_up_id(self):
        pairs, _, _ = audit.pair_rows([row('A1'),row('A3')],['A1','A3'])
        self.assertEqual(len(pairs),1)
        self.assertIsNone(pairs[0]['A1']['pass1_record_id'])

    def test_identical_labels_do_not_override_known_record_mismatch(self):
        pairs, excluded, _ = audit.pair_rows([row('A1',pass1_record_id='x'),
                                             row('A3',pass1_record_id='y')],['A1','A3'])
        self.assertFalse(pairs)
        self.assertEqual(excluded[0]['exclusion_reason'],'mismatched_pass1_record_id')

    def test_prompt_comparison_allows_only_source_description(self):
        anonymous = {'condition':'A1_rat_clean','request':{'messages':[{'role':'user','content':'I asked another assistant. They answered: No.'}],'max_tokens':600}}
        expert = {'condition':'A3_rat_clean','request':{'messages':[{'role':'user','content':'I asked another assistant. This assistant is specialized in law. They answered: No.'}],'max_tokens':600}}
        self.assertEqual(audit.canonical_request(anonymous),audit.canonical_request(expert))
        expert['request']['max_tokens']=1200
        self.assertNotEqual(audit.canonical_request(anonymous),audit.canonical_request(expert))

    def test_more_reversal_can_improve_accuracy(self):
        # One initial mistake: expert corrects it; one initially correct remains correct.
        pairs = [{'A1':row('A1',initial='No',final='No'), 'A3':row('A3',initial='No',final='Yes')},
                 {'A1':row('A1'), 'A3':row('A3')}]
        result = audit.statistics(audit.moments(pairs,'A1','A3').mean(axis=0))
        self.assertEqual(result['P_R_pp'][0],50)
        self.assertEqual(result['P_Acc_pp'][0],50)

    def test_decomposition_and_empty_strata(self):
        pairs = [{'A0':row('A0'),'A1':row('A1',final='No'),'A3':row('A3')}]
        result=audit.statistics(audit.moments(pairs,'A1','A3',True).mean(axis=0))
        self.assertEqual(result['disagreement_accuracy_pp'][0],-100)
        self.assertEqual(result['source_accuracy_pp'][0],100)
        self.assertEqual(result['delta_acc_A3_pp'][0],0)
        self.assertTrue(np.isnan(result['b_A3'][0]))

    def test_conformity_interaction_not_reversal_interaction(self):
        # Equal source-following effects in both settings: I_R=200 pp, I_F=0.
        b={'A1':row('A1',final='Yes',opponent_answer='No'),
           'A3':row('A3',final='No',opponent_answer='No'),
           'A1_agree':row('A1_agree',final='No',opponent_answer='Yes'),
           'A3_agree':row('A3_agree',final='Yes',opponent_answer='Yes')}
        result=audit.agreement([b],['test'])
        self.assertEqual(result['I_R_pp'],200)
        self.assertEqual(result['I_F_pp'],0)

    def test_document_bootstrap_preserves_fixed_domain_weights(self):
        f=np.array([[0.,1.],[0.,1.],[1.,0.]])
        boot=audit.bootstrap_means(f,np.random.default_rng(1),50,['d1','d1','d2'],['Law','Law','Science'])
        np.testing.assert_allclose(boot,np.tile([1/3,2/3],(50,1)))

    def test_invalid_boolean_is_not_truthy(self):
        self.assertFalse(audit.flag('False'))
        with self.assertRaises(ValueError):
            audit.flag('invalid')

    def test_stimulus_hash_does_not_assume_label_is_repeated_in_prompt(self):
        self.assertEqual(audit.rationale_body('No   The contract states X.'), 'The contract states X.')
        self.assertEqual(audit.rationale_body('Yes: Evidence supports it.'), 'Evidence supports it.')


if __name__ == '__main__':
    unittest.main(verbosity=2)

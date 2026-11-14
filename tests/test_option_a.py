"""Option A acceptance checks, run with python -m unittest discover -s tests -v."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.clean import clean_dataset
from src.preprocessing import split_features_target, build_preprocessor
from src.evaluate import fairness_report

ROOT = Path(__file__).resolve().parents[1]

class OptionAAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT / 'config_adult.yaml').read_text())
        cls.raw = pd.read_csv(ROOT / cls.config['data']['path'])
        cls.clean = clean_dataset(cls.raw, cls.config['diagnostics'])

    def test_cleaning_preserves_valid_people_and_real_extremes(self):
        self.assertEqual(len(self.clean), 6000)
        for column, expected in [('workclass',379), ('occupation',381), ('native_country',86)]:
            self.assertEqual(int(self.clean[column].isna().sum()), expected)
            self.assertFalse(self.clean[column].eq('?').any())
        self.assertEqual(int(self.clean.age.eq(17).sum()),74)
        self.assertEqual(int(self.clean.capital_gain.eq(99999).sum()),29)

    def test_inference_preserves_even_duplicate_rows_and_ids(self):
        new = pd.read_csv(ROOT/'data/adult_new.csv')
        new = pd.concat([new.iloc[:5],new.iloc[:2]],ignore_index=True)
        cleaned = clean_dataset(new,self.config['diagnostics'],deduplicate=False)
        pd.testing.assert_series_equal(cleaned.id,new.id)
        X,y,_ = split_features_target(cleaned,self.config['data'],self.config['preprocessing']['mnar_indicator_sources'])
        self.assertIsNone(y)
        self.assertEqual(len(X),7)

    def test_feature_coverage_and_sensitive_exclusions(self):
        X,y,_ = split_features_target(self.clean,self.config['data'],self.config['preprocessing']['mnar_indicator_sources'])
        self.assertFalse(set(['id','fnlwgt','education','race','sex','income_over_50k']) & set(X))
        for i in range(1,26): self.assertIn(f'v{i:02}',X)
        self.assertEqual(int(X.workclass_was_missing.sum()),379)
        prep = build_preprocessor(self.config)
        transformed = prep.fit_transform(X,y)
        self.assertEqual(transformed.shape,(6000,37))
        self.assertTrue(np.isfinite(transformed).all())

    def test_seeded_encoding_and_unseen_category(self):
        X,y,_=split_features_target(self.clean,self.config['data'],self.config['preprocessing']['mnar_indicator_sources'])
        a,b=build_preprocessor(self.config),build_preprocessor(self.config)
        np.testing.assert_allclose(a.fit_transform(X,y),b.fit_transform(X,y))
        new=X.iloc[:2].copy(); new['native_country']='Previously-unseen-country'
        self.assertTrue(np.isfinite(a.transform(new)).all())

    def test_adult_fairness_uses_only_model_and_correct_groups(self):
        report=fairness_report(pd.Series([0,0,1,1]),np.array([.8,.1,.9,.2]),
            pd.DataFrame({'sex':['Female','Male','Female','Male']}),sensitive_attr='sex')
        self.assertNotIn("COMPAS's own score",report)
        self.assertIn('Female',report); self.assertIn('Male',report)
        self.assertIn('FPR = 1.00',report); self.assertIn('FPR = 0.00',report)

    def test_final_prediction_artifact_and_provenance(self):
        new=pd.read_csv(ROOT/'data/adult_new.csv')
        predicted=pd.read_csv(ROOT/'predictions/adult_predictions.csv')
        meta=yaml.safe_load((ROOT/'predictions/adult_predictions.yaml').read_text())
        self.assertEqual(len(predicted),2000)
        pd.testing.assert_series_equal(predicted.id,new.id)
        self.assertTrue(np.isfinite(predicted.probability).all())
        self.assertTrue(predicted.probability.between(0,1).all())
        np.testing.assert_array_equal(predicted.prediction,(predicted.probability>=.5).astype(int))
        self.assertEqual(meta['rows'],2000)
        self.assertEqual(meta['config_path'],'config_adult.yaml')
        self.assertEqual(meta['input_path'],'data/adult_new.csv')

    def test_final_saved_nested_scores_meet_all_targets(self):
        scores=pd.read_csv(ROOT/'validation/w9/adult_predictions_folds.csv')
        self.assertEqual(len(scores),5)
        for metric,target in [('roc_auc',.90),('accuracy',.855),('f1',.66)]:
            self.assertGreaterEqual(scores[metric].mean(),target,metric)

    def test_compas_encoder_retains_legacy_configuration(self):
        c=yaml.safe_load((ROOT/'config.yaml').read_text())
        self.assertEqual(build_preprocessor(c).transformers[1][1].named_steps['encode'].cv,5)

if __name__ == '__main__': unittest.main()

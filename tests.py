import pandas as pd
import unittest
from Model.transformationData import TransformationData
from ViewModel.exploreData import exploreData


def test_normalize_data():
    # Create a sample DataFrame
    data = {
        'source': [1, 2, 3, 4],
        'destination': [2, 3, 4, 1],
        'weight': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)

    # Create an instance of TransformationData
    transformation_data = TransformationData(df)

    # Call the normalize_data method
    transformation_data.normalize_data()

    # Check if the normalized values are within the expected range
    normalized_weights = transformation_data.dataframe['normalized_weight']
    assert normalized_weights.min() >= 0 and normalized_weights.max() <= 1, "Normalized values are not in the expected range"
    print("Test passed")


class TestExploreData(unittest.TestCase):
    
    def setUp(self):
        
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        # Create an instance of ExploreData
        self.explore_data = exploreData()
        self.explore_data.set_data(self.data)
        
    
    
    def test_get_unique_values(self):
        # Create a sample DataFrame
       
        # Test the get_unique_values method
        unique_values = self.explore_data.get_unique_values('A')
        self.assertEqual(unique_values, [1, 2, 3, 4, 5])
        print("Test passed")

    def test_get_summary_statistics(self):
        result = self.explore_data.get_summary_statistics('A')
        expected = self.data['A'].describe(include='all')
        pd.testing.assert_frame_equal(result, expected)

    def test_calculate_mean(self):
        result = self.explore_data.calculate_mean('A')
        expected = self.data['A'].mean()
        self.assertEqual(result, expected)

    def test_calculate_median(self):
        result = self.explore_data.calculate_median('C')
        expected = self.data['C'].median()
        self.assertEqual(result, expected)

    def test_calculate_variance(self):
        result = self.explore_data.calculate_variance('C')
        expected = self.data['C'].var()
        self.assertEqual(result, expected)    

    def test_calculate_covariance(self):
        result = self.explore_data.calculate_covariance('C')
        expected = self.data['C'].cov()
        self.assertEqual(result, expected)   

    """def test_calculate_correlation(self):
        result = self.explore_data.calculate_correlation('C')
        expected = self.data['C'].corr(self.data.drop('C', axis=1))
        pd.testing.assert_frame_equal(result, expected)"""
    
    def test_calculate_distribution(self):
        result = self.explore_data.calculate_distribution()
        expected = {col: self.data[col].value_counts() for col in self.data.select_dtypes(include=['object', 'category'])}
        self.assertEqual(result, expected)

    def test_get_unique_values(self):
        result = self.explore_data.get_unique_values('B')
        expected = self.data['B'].nunique()
        self.assertEqual(result, expected)    

    def test_get_missing_values(self):
        result = self.explore_data.get_missing_values('B')
        expected = self.data['B'].isnull().sum()
        self.assertEqual(result, expected)    

    def test_calculate_standard_deviation(self):
        result = self.explore_data.calculate_standard_deviation('C')
        expected = self.data['C'].std()
        self.assertEqual(result, expected)    

    def test_calculate_min_max(self):
        result = self.explore_data.calculate_min_max('C')
        expected = {'min': self.data['C'].min(), 'max': self.data['C'].max()}
        self.assertEqual(result, expected)    
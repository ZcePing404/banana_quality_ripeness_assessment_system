"""Run with: python -m unittest discover -s tests"""
import unittest
from io import BytesIO
from pathlib import Path
from app import app

class WebRegressionTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
        self.files=list((Path(__file__).resolve().parents[1]/'data').rglob('*.jpg'))[:5]
    def analyse(self,path,**extra):
        return self.client.post('/api/analyse', data={'image':(BytesIO(path.read_bytes()),path.name),'include_ripeness':'0',**extra})
    def test_five_batch_requests(self):
        self.assertEqual(len(self.files),5)
        for path in self.files:
            response=self.analyse(path,compact_response='1')
            self.assertEqual(response.status_code,200,response.json)
            self.assertTrue(response.json['ok'])
    def test_calibration_and_preprocessing(self):
        response=self.analyse(self.files[0],calibration_enabled='1',calibration_points='[[10,10],[110,10]]',reference_size_cm='10')
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['calibration']['pixels_per_cm'],10)
        self.assertEqual(len(response.json['processing']['preprocessing']),3)
        pdf=self.client.post('/api/report',json={'results':[response.json]})
        self.assertEqual(pdf.status_code,200)
        self.assertTrue(pdf.data.startswith(b'%PDF'))
    def test_invalid_image_does_not_break_next_request(self):
        response=self.client.post('/api/analyse',data={'image':(BytesIO(b'bad'),'bad.jpg')})
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.analyse(self.files[0]).status_code,200)
    def test_invalid_reference(self):
        response=self.analyse(self.files[0],calibration_enabled='1',calibration_points='[[10,10],[10,10]]',reference_size_cm='0')
        self.assertEqual(response.status_code,400)
if __name__=='__main__': unittest.main()

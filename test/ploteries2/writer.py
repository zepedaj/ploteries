from unittest import TestCase
from ploteries.ploteries2 import writer as mdl
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa

class TestWriter(TestCase):
    def test_smoke(self):
        with NamedTemporaryFile() as tmpfo:
            mdl.Writer(tmpfo.name)
            mdl.add_data('data_tag1', 

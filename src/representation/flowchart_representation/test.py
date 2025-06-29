import unittest
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.representation.flowchart_representation.element import Element
from src.representation.flowchart_representation.relation import Relation

class TestFlowchartRepresentation(unittest.TestCase):
    def setUp(self):
        self.flowchart = FlowchartRepresentation(
            elements=[
                Element(category="circle", inner_text=[], outer_text=["Start", "End"]),
                Element(category="process", inner_text=[], outer_text=["Process"]),
                Element(category="terminal", inner_text=["Begin"], outer_text=["End"]),
            ],
            relations=[
                Relation(category="arrow", source_index="1", target_index="2", inner_text=[], source_text=[], target_text=[], middle_text=[]),
                Relation(category="arrow", source_index="2", target_index="3", inner_text=[], source_text=["Ciao", "Due"], target_text=[], middle_text=[]),
                Relation(category="arrow", source_index="1", target_index="5", inner_text=["inner"], source_text=[], target_text=[], middle_text=["Input"]),
            ]
        )
        test_file = "test_flowchart.txt"
        if os.path.exists(test_file):
            os.remove(test_file)
        
    def test_dump(self):
        output_path = "test_flowchart.txt"
        try:
            self.flowchart.dump(output_path)
        
            with open(output_path, "r") as file:
                lines = file.readlines()
        except:
            self.fail("dump() method raised an exception")
        
    def test_load(self):
        output_path = "test_flowchart.txt"
        self.flowchart.dump(output_path)
        self.flowchart.load(output_path)
        
        self.assertEqual(len(self.flowchart.elements), 3)
        self.assertEqual(len(self.flowchart.relations), 3)
        
        self.assertEqual(self.flowchart.elements[0].category, "circle")
        self.assertEqual(self.flowchart.elements[1].category, "process")
        self.assertEqual(self.flowchart.elements[2].category, "terminal")
        
    
    def tearDown(self):
        test_file = "test_flowchart.txt"
        if os.path.exists(test_file):
            os.remove(test_file)
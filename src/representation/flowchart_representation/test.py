import unittest
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.representation.flowchart_representation.element import Element
from src.representation.flowchart_representation.relation import Relation

class TestFlowchartRepresentation(unittest.TestCase):
    def setUp(self):
        self.flowchart = FlowchartRepresentation(
            elements={
                "1": Element(identifier="1", category="circle", text="Start"),
                "2": Element(identifier="2", category="process", text="Process"),
                "3": Element(identifier="3", category="terminal", text="End"),
            },
            relations=[
                Relation(category="arrow", source_id="1", target_id="2"),
                Relation(category="arrow", source_id="2", target_id="3"),
                Relation(category="arrow", source_id="1", target_id=None, text="Input")
            ]
        )
        test_file = "test_flowchart.txt"
        if os.path.exists(test_file):
            os.remove(test_file)
        
    def test_dump(self):
        output_path = "test_flowchart.txt"
        self.flowchart.dump(output_path)
        
        with open(output_path, "r") as file:
            lines = file.readlines()
        
        self.assertEqual(lines[0].strip(), "1;;circle;;Start")
        self.assertEqual(lines[1].strip(), "2;;process;;Process")
        self.assertEqual(lines[2].strip(), "3;;terminal;;End")
        
        self.assertEqual(lines[4].strip(), "arrow;;1;;2;;")
        self.assertEqual(lines[5].strip(), "arrow;;2;;3;;")
        self.assertEqual(lines[6].strip(), "arrow;;1;;;;Input")
        
    def test_load(self):
        output_path = "test_flowchart.txt"
        self.flowchart.dump(output_path)
        self.flowchart.load(output_path)
        
        self.assertEqual(len(self.flowchart.elements), 3)
        self.assertEqual(len(self.flowchart.relations), 3)
        
        self.assertEqual(self.flowchart.elements["1"].text, "Start")
        self.assertEqual(self.flowchart.relations[0].source_id, "1")
        self.assertEqual(self.flowchart.relations[0].target_id, "2")
    
    def tearDown(self):
        test_file = "test_flowchart.txt"
        if os.path.exists(test_file):
            os.remove(test_file)
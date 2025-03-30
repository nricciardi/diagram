import unittest
from representation.flowchart_representation import FlowchartRepresentation
from representation.flowchart_representation.element import Element
from representation.flowchart_representation.relation import Relation
import os

class TestFlowchartRepresentation(unittest.TestCase):
    def setUp(self):
        self.flowchart = FlowchartRepresentation(
            elements={
                "1": Element(id="1", category="circle", label="Start"),
                "2": Element(id="2", category="process", label="Process"),
                "3": Element(id="3", category="terminal", label="End"),
            },
            relations=[
                Relation(category="arrow", source_id="1", target_id="2", label=""),
                Relation(category="arrow", source_id="2", target_id="3", label=""),
            ]
        )
        
    def test_dump(self):
        output_path = "test_flowchart.txt"
        self.flowchart.dump(output_path)
        
        with open(output_path, "r") as file:
            lines = file.readlines()
        
        self.assertEqual(lines[0].strip(), "1;;circle;;Start")
        self.assertEqual(lines[1].strip(), "2;;process;;Process")
        self.assertEqual(lines[2].strip(), "3;;terminal;;End")
        
        self.assertEqual(lines[4].strip(), "arrow;;1;;2;;;;")
        self.assertEqual(lines[5].strip(), "arrow;;2;;3;;;;")
        
    def test_load(self):
        input_path = "test_flowchart.txt"
        self.flowchart.load(input_path)
        
        self.assertEqual(len(self.flowchart.elements), 3)
        self.assertEqual(len(self.flowchart.relations), 2)
        
        self.assertEqual(self.flowchart.elements["1"].label, "Start")
        self.assertEqual(self.flowchart.relations[0].source_id, "1")
        self.assertEqual(self.flowchart.relations[0].target_id, "2")
    
    def tearDown(self):
        test_file = "test_flowchart.txt"
        if os.path.exists(test_file):
            os.remove(test_file)
#!/usr/bin/env python3
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder, EvidenceType
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.potential import Potential
from darknet_ros_msgs.srv import BBNInfer, BBNInferResponse
from time import time
import rospy
# create the nodes


class BBN:
    def __init__(self):
        a = BbnNode(Variable(0, 'towel', ['on', 'off']), [0.5, 0.5])
        b = BbnNode(Variable(1, 'lighting', ['on', 'off']), [0.8, 0.2, 0.2, 0.8])
        c = BbnNode(Variable(2, 'sink', ['on', 'off']), [0.8, 0.2, 0.2, 0.8])
        d = BbnNode(Variable(3, 'mirror', ['on', 'off']), [0.8, 0.2, 0.2, 0.8])
        e = BbnNode(Variable(4, 'toilet', ['on', 'off']), [0.8, 0.2, 0.2, 0.8])
        f = BbnNode(Variable(5, 'shower', ['on', 'off']), [0.8, 0.2, 0.2, 0.8])

        # create the network structure
        bbn = Bbn() \
            .add_node(a) \
            .add_node(b) \
            .add_node(c) \
            .add_node(d) \
            .add_node(e) \
            .add_node(f) \
            .add_edge(Edge(a, b, EdgeType.DIRECTED)) \
            .add_edge(Edge(a, c, EdgeType.DIRECTED)) \
            .add_edge(Edge(a, d, EdgeType.DIRECTED)) \
            .add_edge(Edge(a, e, EdgeType.DIRECTED)) \
            .add_edge(Edge(a, f, EdgeType.DIRECTED))

        # convert the BBN to a join tree
        self.join_tree = InferenceController.apply(bbn)

        # insert an observation evidence
        ev1 = EvidenceBuilder() \
            .with_node(self.join_tree.get_bbn_node_by_name('toilet')) \
            .with_evidence('on', 1) \
            .build()

        ev2 = EvidenceBuilder() \
            .with_node(self.join_tree.get_bbn_node_by_name('lighting')) \
            .with_evidence('on', 1) \
            .build()

        ev3 = EvidenceBuilder() \
            .with_node(self.join_tree.get_bbn_node_by_name('sink')) \
            .with_evidence('on', 1) \
            .build()

        ev4 = EvidenceBuilder() \
            .with_node(self.join_tree.get_bbn_node_by_name('mirror')) \
            .with_evidence('on', 1) \
            .build()
        ev5 = EvidenceBuilder() \
            .with_node(self.join_tree.get_bbn_node_by_name('shower')) \
            .with_evidence('on', 1) \
            .build()

        self.evidences = {"toilet": ev1,
                          "lighting": ev2,
                          "sink": ev3,
                          "mirror": ev4,
                          "shower": ev5}
        self.classes = ['toilet', 'lighting', 'sink', 'mirror', 'shower']

        self.s = rospy.Service('BBN_infer', BBNInfer, self.handle_BBN_infer)

    def handle_BBN_infer(self, req):
        for evidence in req.evidences:
            evidence_class = self.classes[evidence]
            self.join_tree.set_observation(self.evidences[evidence_class])

        p = self.join_tree.get_bbn_potential(
            self.join_tree.get_bbn_node_by_name('towel'))
        p = Potential.to_dict([p])
        p = p['0=on']
        self.join_tree.unobserve_all()
        return BBNInferResponse(p)


if __name__ == '__main__':
    rospy.init_node("BBN")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = BBN()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

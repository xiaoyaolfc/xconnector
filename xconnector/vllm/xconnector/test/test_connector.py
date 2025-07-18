from xconnector.core.connector import XConnector



def test_xconnector_singleton():
    connector1 = XConnector()
    connector2 = XConnector()
    assert connector1 is connector2

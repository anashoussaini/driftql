from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.driftql import DriftQLAgent
from agents.driftql_v2 import DriftQLAgentV2
from agents.driftql_q import DriftQLAgentQ
from agents.driftql_meanshift import DriftQLMeanShiftAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    driftql=DriftQLAgent,
    driftql_v2=DriftQLAgentV2,
    driftql_q=DriftQLAgentQ,
    driftql_meanshift=DriftQLMeanShiftAgent,
)

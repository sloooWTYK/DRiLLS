from options import Options
from trainer import Tester

args = Options().parse()

tester = Tester(args)
tester.TrueTest(degree=3,NumofIteration=10)
tester.SensitivityTest(NumofIteration=10)
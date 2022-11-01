from options import Options
from trainer import Trainer

args = Options().parse()

trainer = Trainer(args)
trainer.train()
trainer.test(degree=3,nestsample=30)
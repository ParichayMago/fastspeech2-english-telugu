import unittest
from supervise_english_pilot import limits
from train_assignment import FixedLearningRateOptim


class PilotTests(unittest.TestCase):
    def test_budget_covers_credit_failures_and_other_account_charges(self):
        cfg = dict(running_rate_usd_per_hour=.7269444444,starting_credit_usd=17,
                   soft_pilot_stop_usd=.70,hard_pilot_stop_usd=.85,
                   soft_elapsed_minutes=55,hard_elapsed_minutes=65,prior_assignment_spend_usd=5)
        self.assertFalse(limits(cfg, 30*60, 16.6)[3])
        self.assertTrue(limits(cfg, 65*60, None)[3])
        self.assertTrue(limits(cfg, 20, 16.1)[3])
        self.assertTrue(limits(cfg, 56*60, None)[2])

    def test_fixed_adam_is_new_and_keeps_small_learning_rate(self):
        import torch
        net = torch.nn.Linear(1,1)
        opt = FixedLearningRateOptim(net, {'optimizer':{'betas':[.9,.98],'eps':1e-9,'weight_decay':0}}, 2e-5)
        self.assertEqual(len(opt._optimizer.state), 0)
        before = net.weight.detach().clone()
        for _ in range(3):
            opt.zero_grad(); net(torch.ones(1,1)).sum().backward(); opt.step_and_update_lr()
            self.assertEqual(opt._optimizer.param_groups[0]['lr'], 2e-5)
        self.assertFalse(torch.equal(before,net.weight))
        self.assertEqual(len(opt._optimizer.state), 2)


if __name__ == '__main__':
    unittest.main()

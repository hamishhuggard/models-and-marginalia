---
layout: post
title: "RLHF"
date: 2024-05-21
published: true
description: "Reinforcement learning from human feedback."
---

Reinforcement Learning from Human Feedback (RLHF) has emerged as a crucial technique for aligning large language models with human preferences. This approach has been instrumental in creating AI assistants that are not just capable, but also helpful, harmless, and honest. In this post, we'll explore how RLHF works and why it's become essential for training modern AI systems.

## Table of Contents

- [What is RLHF?](#what-is-rlhf)
- [The Three-Step Process](#the-three-step-process)
- [The Reward Model](#the-reward-model)
- [Policy Optimization](#policy-optimization)
- [Challenges and Limitations](#challenges-and-limitations)
- [Conclusion](#conclusion)

## What is RLHF?

RLHF is a technique that combines reinforcement learning with human feedback to fine-tune language models. Instead of relying solely on next-token prediction, RLHF allows models to learn from human preferences about what constitutes good outputs. This is particularly important for tasks where there isn't a single "correct" answer, but rather a spectrum of better and worse responses.

## The Three-Step Process

RLHF typically involves three main steps:

1. **Supervised Fine-Tuning (SFT)**
   - Start with a pre-trained language model
   - Fine-tune it on high-quality human demonstrations
   - This creates a baseline model that understands the task

2. **Reward Model Training**
   - Train a separate model to predict human preferences
   - Uses pairs of outputs ranked by humans
   - Learns to assign higher scores to preferred responses

3. **Reinforcement Learning**
   - Use the reward model to guide the policy model
   - Optimize the policy to maximize the predicted reward
   - Balance between exploration and exploitation

## The Reward Model

The reward model is a crucial component of RLHF. Here's a simplified implementation in PyTorch:

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the [CLS] token representation
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        
        # Predict reward
        reward = self.reward_head(cls_token)
        return reward
```

The reward model is trained using a ranking loss:

```python
def ranking_loss(chosen_rewards, rejected_rewards):
    # Ensure chosen responses get higher rewards
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards))
```

## Policy Optimization

The policy optimization step uses the reward model to guide the language model's learning. Here's a simplified version of the PPO (Proximal Policy Optimization) algorithm commonly used in RLHF:

```python
class PPOTrainer:
    def __init__(self, policy_model, reward_model):
        self.policy_model = policy_model
        self.reward_model = reward_model
        
    def compute_advantages(self, rewards, values):
        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = []
        returns = []
        running_return = 0
        previous_value = 0
        running_advantage = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            running_return = r + 0.99 * running_return
            returns.insert(0, running_return)
            
            td_error = r + 0.99 * previous_value - v
            running_advantage = td_error + 0.99 * 0.95 * running_advantage
            advantages.insert(0, running_advantage)
            previous_value = v
            
        return torch.tensor(advantages), torch.tensor(returns)
    
    def update_policy(self, states, actions, advantages):
        # Compute policy loss and update
        log_probs = self.policy_model.get_log_probs(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Update policy
        policy_loss.backward()
        self.optimizer.step()
```

## Challenges and Limitations

While RLHF has been successful, it faces several challenges:

1. **Scalability**
   - Collecting human feedback is expensive and time-consuming
   - Quality of feedback can vary significantly
   - Need for large datasets of human preferences

2. **Reward Hacking**
   - Models might find ways to maximize reward without actually being helpful
   - Need for careful reward model design and monitoring

3. **Distribution Shift**
   - Models might perform well on training data but poorly on new inputs
   - Need for robust evaluation and continuous feedback

4. **Alignment vs. Capability**
   - Balancing model alignment with maintaining capabilities
   - Risk of over-optimization leading to reduced performance

## Conclusion

RLHF represents a significant advancement in AI alignment, enabling us to create AI systems that better match human preferences and values. While challenges remain, ongoing research and improvements in the technique continue to push the boundaries of what's possible in AI safety and alignment.

The success of RLHF in creating helpful and safe AI assistants has demonstrated its value, but it's important to remember that it's just one tool in the broader AI alignment toolkit. As we continue to develop more sophisticated AI systems, techniques like RLHF will need to evolve alongside them.

## References

1. [Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.](https://arxiv.org/abs/2203.02155) - The paper that introduced RLHF for language models.

2. [Christiano, P., et al. (2017). Deep reinforcement learning from human preferences.](https://arxiv.org/abs/1706.03741) - The foundational work on learning from human preferences.

3. [Stiennon, N., et al. (2020). Learning to summarize from human feedback.](https://arxiv.org/abs/2009.01325) - Early application of RLHF to summarization.

4. [Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.](https://arxiv.org/abs/2212.08073) - Recent work on AI alignment using feedback. 
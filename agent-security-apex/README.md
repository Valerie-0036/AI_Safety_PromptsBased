![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-security-apex--readme)

# Agent Security Evaluation Tutorial

## Overview

This tutorial teaches prompt injection attacks and defenses through hands-on security testing of AI systems. Instead of just reading about vulnerabilities, you'll actually exploit them and then build protections against them.

## What's included

**Attack techniques**: Learn 8 major types of prompt injection attacks, from direct instruction override to sophisticated encoding-based bypasses.

**Practical testing**: Use real attack datasets and automated testing tools to evaluate AI system security. Includes 91 documented attack examples from security research.

**Defense implementation**: Build and validate security measures using advanced prompt engineering techniques that work in production systems.

**Encoding tools**: Test 12 different obfuscation methods that attackers use to bypass AI filters (Base64, hex, ciphers, etc.).

## What you'll learn

- How to identify and execute prompt injection attacks
- Automated security testing for AI applications  
- Defensive prompt engineering with quantitative validation
- Real-world attack patterns and mitigation strategies

## Prerequisites

- Basic Python programming
- OpenAI API access (you'll need an API key)
- Understanding of AI/ML fundamentals

## Files included

- `agent-security-evaluation-tutorial.ipynb` - Main tutorial notebook
- `model_testing_tools.py` - Automated testing framework
- `prompt_manipulation_tools.py` - Encoding/obfuscation utilities
- `system_prompt.txt` - Example defensive prompt
- `example_prompts.csv` - Dataset of 91 real attack examples

## Setup

1. Install required packages: `pip install openai python-dotenv pandas`
2. Create a `.env` file with your OpenAI API key: `OPENAI_API_KEY=your_key_here`
3. Run the notebook cells in order

The tutorial takes about 30-45 minutes to complete and includes both automated testing and manual experimentation.

## Warning

This tutorial demonstrates actual attack techniques for educational purposes. Use these methods only on systems you own or have explicit permission to test.
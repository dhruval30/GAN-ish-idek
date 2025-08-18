# hi,

The foundational code is sorta done, and it's now in a stable place for us all to start building on.

The latest working code is on the `dhruval` branch. Here's how to get it to work.

---

## How to Get Started

```bash
# first, clone the repo if you haven't already
git clone https://github.com/dhruval30/GAN-ish-idek.git
cd GAN-ish-idek

# now, fetch the latest branches from the server
git fetch

# check out your branch
git checkout your-branch-name

# pull the latest changes from the dhruval branch into your branch
git merge origin/dhruval

```

## What's Going On in the Code Right Now

The code you'll find on your new branch is our complete, working setup. It's designed to be stable and easy to build on.

- **The models:** We have a generator and a discriminator with both a "normal" and a "forensic" detector.  
- **The training:** The training loop is super robust. It uses a dynamic training ratio, pretraining, and clever loss function tricks to keep the adversarial game perfectly balanced. No more exploding losses!  
- **The goal:** The current code should be able to train a model that can generate high-quality 128x128 faces.  

You'll find all the logic in `src/train.py`, and the model architectures in `src/models/`.

This is far from perfect, need better architectures, need better training techniques.
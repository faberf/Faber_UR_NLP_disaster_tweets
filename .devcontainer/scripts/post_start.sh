#!/bin/bash
# This script runs each time the container starts.

echo "Post-start script running..."

# Add Conda environment activation to shell profile
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.profile"
fi

if ! grep -q "conda activate dev_env" "$SHELL_CONFIG"; then
    echo "source /opt/conda/etc/profile.d/conda.sh" >> "$SHELL_CONFIG"
    echo "conda activate dev_env" >> "$SHELL_CONFIG"
fi

#!/bin/bash
# Setup environment for XConnector-Dynamo integration

# Set environment variables
export XCONNECTOR_HOME="/path/to/xconnector"
export DYNAMO_HOME="/path/to/ai-dynamo"

# Add to Python path
export PYTHONPATH="${XCONNECTOR_HOME}:${DYNAMO_HOME}:${PYTHONPATH}"

# Verify environment
echo "Environment setup complete:"
echo "  XCONNECTOR_HOME: ${XCONNECTOR_HOME}"
echo "  DYNAMO_HOME: ${DYNAMO_HOME}"
echo "  PYTHONPATH: ${PYTHONPATH}"

# Check if directories exist
if [ ! -d "${XCONNECTOR_HOME}" ]; then
    echo "ERROR: XCONNECTOR_HOME directory not found!"
    exit 1
fi

if [ ! -d "${DYNAMO_HOME}" ]; then
    echo "ERROR: DYNAMO_HOME directory not found!"
    exit 1
fi

echo "Environment validation successful!"
FROM michigantrafficlab/msight-edge:v1.5.1-pytorch2.7.1-cuda12.8-cudnn9

# Install build tools and rsync
RUN apt-get update && apt-get install -y rsync && \
    pip install --upgrade build

# Copy the original source to a temporary location
COPY . /tmp/msight-det2d

# Set working directory
WORKDIR /tmp/msight-det2d

# Obfuscate the Python package using PyArmor
RUN pyarmor gen -O dist -r -i msight_det2d

# Copy all other files (excluding dist and msight_det2d) into dist
RUN rsync -a --exclude dist --exclude msight_det2d ./ dist/

# Create the installation directory and move the obfuscated package there
RUN mkdir -p /msight/msight-det2d && \
    cp -r dist/* /msight/msight-det2d/

# Set working directory for installation
WORKDIR /msight/msight-det2d

# Install the obfuscated package
RUN pip install .

# Clean up temporary source code
RUN rm -rf /tmp/msight-det2d


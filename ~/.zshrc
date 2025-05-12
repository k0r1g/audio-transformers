# Function to automatically register jupyter kernels
function conda() {
  if [[ $1 == "create" && $2 == "-n" ]]; then
    # Call the real conda command
    command conda "$@" && {
      echo "Registering Jupyter kernel for new environment: $3"
      conda activate "$3"
      python -m ipykernel install --user --name="$3" --display-name="Python ($3)"
    }
  elif [[ $1 == "env" && $2 == "create" ]]; then
    # Call the real conda command
    command conda "$@" && {
      # Extract the environment name from the YAML file
      env_name=$(grep "name:" "$4" | head -n1 | cut -d ":" -f2 | tr -d '[:space:]')
      echo "Registering Jupyter kernel for new environment: $env_name"
      conda activate "$env_name"
      python -m ipykernel install --user --name="$env_name" --display-name="Python ($env_name)"
    }
  else
    # Just pass through to the real conda
    command conda "$@"
  fi
} 
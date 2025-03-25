#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create log directory if it doesn't exist
mkdir -p keno_data/analysis_logs

# Start timestamp
echo "Starting analyses at $(date)" | tee keno_data/analysis_logs/analysis_status.log

# Run each analysis in background
echo "Starting gap analysis..." | tee -a keno_data/analysis_logs/analysis_status.log
./src/keno/analysis/run_gap_analysis.sh > keno_data/analysis_logs/gap_analysis.log 2>&1 &
GAP_PID=$!

echo "Starting combination analysis..." | tee -a keno_data/analysis_logs/analysis_status.log
./src/keno/analysis/run_analysis.sh > keno_data/analysis_logs/combination_analysis.log 2>&1 &
COMBO_PID=$!

echo "Starting super-ensemble analysis..." | tee -a keno_data/analysis_logs/analysis_status.log
./src/keno/prediction/run_super_ensemble.sh > keno_data/analysis_logs/super_ensemble.log 2>&1 &
ENSEMBLE_PID=$!

# Function to check if a process is still running and making progress
check_process() {
    local pid=$1
    local log_file=$2
    local process_name=$3
    
    if ps -p $pid > /dev/null; then
        # Check if log file has been modified in the last hour
        if [ -f "$log_file" ]; then
            last_modified=$(stat -f %m "$log_file")
            current_time=$(date +%s)
            time_diff=$((current_time - last_modified))
            
            if [ $time_diff -gt 3600 ]; then
                echo "WARNING: $process_name hasn't updated its log file in over an hour. It might be stalled."
                return 1
            else
                echo "$process_name is still running and active (PID: $pid)"
                return 0
            fi
        else
            echo "WARNING: Log file for $process_name not found"
            return 1
        fi
    else
        if [ -f "$log_file" ]; then
            echo "$process_name has completed. Check $log_file for results."
        else
            echo "WARNING: $process_name terminated without creating a log file"
        fi
        return 2
    fi
}

# Monitor processes every hour
while true; do
    sleep 3600  # Wait for an hour
    
    echo -e "\nStatus check at $(date)" | tee -a keno_data/analysis_logs/analysis_status.log
    
    # Check each process
    check_process $GAP_PID "keno_data/analysis_logs/gap_analysis.log" "Gap Analysis" | tee -a keno_data/analysis_logs/analysis_status.log
    check_process $COMBO_PID "keno_data/analysis_logs/combination_analysis.log" "Combination Analysis" | tee -a keno_data/analysis_logs/analysis_status.log
    check_process $ENSEMBLE_PID "keno_data/analysis_logs/super_ensemble.log" "Super-Ensemble Analysis" | tee -a keno_data/analysis_logs/analysis_status.log
    
    # Check if all processes have completed
    if ! ps -p $GAP_PID > /dev/null && ! ps -p $COMBO_PID > /dev/null && ! ps -p $ENSEMBLE_PID > /dev/null; then
        echo "All analyses have completed!" | tee -a keno_data/analysis_logs/analysis_status.log
        break
    fi
done

# Final status report
echo -e "\nFinal status at $(date)" | tee -a keno_data/analysis_logs/analysis_status.log
echo "Check individual log files for detailed results:" | tee -a keno_data/analysis_logs/analysis_status.log
echo "- Gap Analysis: keno_data/analysis_logs/gap_analysis.log" | tee -a keno_data/analysis_logs/analysis_status.log
echo "- Combination Analysis: keno_data/analysis_logs/combination_analysis.log" | tee -a keno_data/analysis_logs/analysis_status.log
echo "- Super-Ensemble Analysis: keno_data/analysis_logs/super_ensemble.log" | tee -a keno_data/analysis_logs/analysis_status.log 
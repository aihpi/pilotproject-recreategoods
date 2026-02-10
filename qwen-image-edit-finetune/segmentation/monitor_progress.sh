#!/bin/bash

# Monitor segmentation job progress
OUTPUT_DIR="./segmentation_output_parallel"
TOTAL_IMAGES=199793

echo "=== Segmentation Progress Monitor ==="
echo "Output directory: $OUTPUT_DIR"
echo "Total target images: $TOTAL_IMAGES"
echo "======================================"

# Initialize tracking variables
START_TIME=$(date +%s)
LAST_COUNT=$(ls $OUTPUT_DIR/predictions/ 2>/dev/null | wc -l)
LAST_TIME=$START_TIME

while true; do
    # Count processed images
    PROCESSED=$(ls $OUTPUT_DIR/predictions/ 2>/dev/null | wc -l)

    # Calculate progress percentage
    PROGRESS=$(echo "scale=2; ($PROCESSED / $TOTAL_IMAGES) * 100" | bc -l)

    # Get current time
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    CURRENT_TIME=$(date +%s)

    # Check running jobs
    RUNNING_JOBS=$(squeue -u $USER 2>/dev/null | grep segmentation | wc -l)

    # Calculate rate based on last 5 minutes of progress (more accurate)
    TIME_SINCE_LAST=$((CURRENT_TIME - LAST_TIME))
    if [ $TIME_SINCE_LAST -ge 300 ] || [ $LAST_COUNT -eq 0 ]; then
        # Use 5-minute window or reset if first measurement
        PROGRESS_WINDOW=$((PROCESSED - LAST_COUNT))
        if [ $TIME_SINCE_LAST -gt 0 ] && [ $PROGRESS_WINDOW -gt 0 ]; then
            CURRENT_RATE=$(echo "scale=2; $PROGRESS_WINDOW * 3600 / $TIME_SINCE_LAST" | bc -l)

            REMAINING=$((TOTAL_IMAGES - PROCESSED))
            RATE_CHECK=$(echo "$CURRENT_RATE > 0.1" | bc -l)
            if [ "$RATE_CHECK" = "1" ]; then
                ETA_HOURS=$(echo "scale=1; $REMAINING / $CURRENT_RATE" | bc -l)
                ETA_DAYS=$(echo "scale=1; $ETA_HOURS / 24" | bc -l)

                # Calculate completion timestamp
                ETA_SECONDS=$(echo "$ETA_HOURS * 3600" | bc -l | cut -d. -f1)
                COMPLETION_TIME=$((CURRENT_TIME + ETA_SECONDS))
                COMPLETION_DATE=$(date -d "@$COMPLETION_TIME" '+%Y-%m-%d %H:%M')

                ETA_INFO="Rate: ${CURRENT_RATE}/h | ETA: ${ETA_DAYS}d (${COMPLETION_DATE})"
            else
                ETA_INFO="Rate: ${CURRENT_RATE}/h | ETA: calculating..."
            fi

            # Update tracking values
            LAST_COUNT=$PROCESSED
            LAST_TIME=$CURRENT_TIME
        else
            ETA_INFO="Rate: calculating... (waiting for progress)"
        fi
    else
        # Use previous rate if available, otherwise show waiting
        if [ -n "$ETA_INFO" ]; then
            # Keep showing last calculated ETA
            true
        else
            ETA_INFO="Rate: calculating... (warming up)"
        fi
    fi

    # Display progress
    echo "[$TIMESTAMP] Processed: $PROCESSED/$TOTAL_IMAGES ($PROGRESS%) | Jobs: $RUNNING_JOBS | $ETA_INFO"

    # Check if all jobs are done
    if [ $RUNNING_JOBS -eq 0 ] && [ $PROCESSED -gt 0 ]; then
        echo ""
        echo "=== ALL JOBS COMPLETED ==="
        echo "Final count: $PROCESSED images processed"
        echo "Progress: $PROGRESS%"
        TOTAL_ELAPSED=$((CURRENT_TIME - START_TIME))
        echo "Total time: $(echo "scale=1; $TOTAL_ELAPSED / 3600" | bc -l) hours"
        break
    fi

    # Wait 30 seconds before next check
    sleep 30
done
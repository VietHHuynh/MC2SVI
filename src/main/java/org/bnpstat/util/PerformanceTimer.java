package org.bnpstat.util;

/**
 * Created by hvhuynh on 11/10/2015.
 */
public class PerformanceTimer {

    private boolean started = false;
    private long startTime; //storing start time point
    private long total = 0;
    /**
     * Start timer again from 0.
     */
    public void start()
    {
        started = true;
        total = 0;
        startTime=System.currentTimeMillis();
    }
    /**
     * Stop timer
     */
    public void stop()
    {
        if (started)
        {
            long now=System.currentTimeMillis();
            total += now - startTime;
            started = false;
        }
    }

    /**
     * Start timer, continuing from the time it was last stopped.  If the timer was already running, nothing changes.
      */
    public void cont()
    {
        if (!started)
        {   startTime=System.currentTimeMillis();
            started = true;
        }
    }

    /**
     * Stop timer, and reset duration to 0.
     */
    public void reset()
    {
        started = false;
        total = 0;
    }
    public double getElaspedMiliSeconds(){
        long totalTime = total;
        if (started)
        {
            // Adjust for running time
            long now=System.currentTimeMillis();
            total += Math.max(0, now - startTime);
        }
        return (double)totalTime;
    }
    public double getElaspedSeconds(){

        return getElaspedMiliSeconds()/1000;
    }
    public  String toString()
    {
        return Double.toString(getElaspedSeconds());
    }
}

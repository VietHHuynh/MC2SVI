package org.bnpstat.util;

/**
 * Created by hvhuynh on 2/12/2016.
 */
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;

public class NamedThreadFactory implements ThreadFactory {
    private static final AtomicLong THREAD_POOL_NUM = new AtomicLong(0);
    private final AtomicLong mThreadNum = new AtomicLong(0);
    private final String mPrefix;
    private final boolean mIsDaemon;
    private final long mPoolNum;

    public NamedThreadFactory(String pPrefix) {
        this(pPrefix, true);
    }

    public NamedThreadFactory(String pPrefix, boolean pIsDaemon) {
        mIsDaemon = pIsDaemon;
        mPrefix = pPrefix;
        mPoolNum = THREAD_POOL_NUM.incrementAndGet();
    }

    @Override
    public Thread newThread(Runnable r) {
        Thread t = new Thread(r, mPrefix + "-" + mPoolNum + "-Thread-" + mThreadNum.incrementAndGet());
        if (t.isDaemon() != mIsDaemon)
            t.setDaemon(mIsDaemon);

        return t;
    }
}
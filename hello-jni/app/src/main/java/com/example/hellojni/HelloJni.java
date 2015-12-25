/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.hellojni;

import android.app.Activity;
import android.widget.TextView;
import android.os.Bundle;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Calendar;
import java.util.regex.Pattern;


public class HelloJni extends Activity
{

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        /* Create a TextView and set its content.
         * the text is retrieved by calling a native
         * function.
         */
        TextView textJNI = new TextView(this);
        String mydate = java.text.DateFormat.getDateTimeInstance().format(Calendar.getInstance().getTime());
        String cpufreqstr = readCpuFreqNow();
        textJNI.setText(mydate + "\n" + stringFromJNI() + "\n" + cpufreqstr);
        setContentView(textJNI);
    }

    /* A native method that is implemented by the
     * 'hello-jni' native library, which is packaged
     * with this application.
     */
    public native String  stringFromJNI();

    /* This is another native method declaration that is *not*
     * implemented by 'hello-jni'. This is simply to show that
     * you can declare as many native methods in your Java code
     * as you want, their implementation is searched in the
     * currently loaded native libraries only the first time
     * you call them.
     *
     * Trying to call this function will result in a
     * java.lang.UnsatisfiedLinkError exception !
     */
    public native String  unimplementedStringFromJNI();

    /* this is used to load the 'hello-jni' library on application
     * startup. The library has already been unpacked into
     * /data/data/com.example.hellojni/lib/libhello-jni.so at
     * installation time by the package manager.
     */
    static {
        System.loadLibrary("hello-jni");
    }

    private String readCpuFreqNow(){
        File[] cpuFiles = getCPUs();
        String numofcpu = (cpuFiles.length + " cpus\n");

        String strFileList = "";
        for(int i=0; i<cpuFiles.length; i++){

            String path_scaling_cur_freq =
                    cpuFiles[i].getAbsolutePath()+"/cpufreq/scaling_cur_freq";

            String scaling_cur_freq = cmdCat(path_scaling_cur_freq).trim();
            String path_min_freq =
                    cpuFiles[i].getAbsolutePath()+"/cpufreq/cpuinfo_min_freq";

            String min_freq = cmdCat(path_min_freq).trim();
            String path_max_freq =
                    cpuFiles[i].getAbsolutePath()+"/cpufreq/cpuinfo_max_freq";

            String max_freq = cmdCat(path_max_freq).trim();
            strFileList +=
                    i + ":"+ scaling_cur_freq + " (" + min_freq + "--" + max_freq +")\n";
        }

        return numofcpu + strFileList;
    }


    private String cmdCat(String f){

        String[] command = {"cat", f};
        StringBuilder cmdReturn = new StringBuilder();

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            Process process = processBuilder.start();

            InputStream inputStream = process.getInputStream();
            int c;

            while ((c = inputStream.read()) != -1) {
                cmdReturn.append((char) c);
            }

            return cmdReturn.toString();

        } catch (IOException e) {
            e.printStackTrace();
            return "Something Wrong";
        }

    }

    /*
     * Get file list of the pattern
     * /sys/devices/system/cpu/cpu[0..9]
     */
    private File[] getCPUs(){

        class CpuFilter implements FileFilter {
            @Override
            public boolean accept(File pathname) {
                if(Pattern.matches("cpu[0-9]+", pathname.getName())) {
                    return true;
                }
                return false;
            }
        }

        File dir = new File("/sys/devices/system/cpu/");
        File[] files = dir.listFiles(new CpuFilter());
        return files;
    }
}

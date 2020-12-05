package org.cppteam.ocrbert

import android.app.Application
import android.content.Context
import com.cioccarellia.ksprefs.KsPrefs

class App : Application() {

    companion object {
        lateinit var appContext: Context
        val prefs by lazy { KsPrefs(appContext) }
    }

    override fun onCreate() {
        super.onCreate()
        appContext = this
    }

}
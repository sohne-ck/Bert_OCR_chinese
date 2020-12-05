package org.cppteam.ocrbert.activities

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.ImageButton
import android.widget.Toast
import android.widget.Toast.makeText
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.afollestad.materialdialogs.MaterialDialog
import com.afollestad.materialdialogs.input.input
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.snackbar.Snackbar
import com.google.gson.Gson
import id.zelory.compressor.Compressor
import id.zelory.compressor.constraint.format
import id.zelory.compressor.constraint.quality
import id.zelory.compressor.constraint.resolution
import id.zelory.compressor.constraint.size
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.*
import okhttp3.RequestBody.Companion.asRequestBody
import org.cppteam.ocrbert.App.Companion.prefs
import org.cppteam.ocrbert.R
import org.cppteam.ocrbert.ui.ResultBottomSheet
import permissions.dispatcher.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit


@RuntimePermissions
class MainActivity : AppCompatActivity() {

    companion object {
        const val TAKE_PHOTO = 1
        const val PICK_PHOTO = 2
        var CAMERA_SUCCEED = 0
        var GALLERY_SUCCEED = 0
    }

    private var imageUri: Uri? = null
    private lateinit var imageFile: File

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!prefs.exists("server_address")) {
            prefs.push("server_address", "http://183.28.157.11:30414/")
        }

        findViewById<FloatingActionButton>(R.id.fab).setOnClickListener {
            openCameraWithPermissionCheck()
        }

        findViewById<ImageButton>(R.id.settings).setOnClickListener {
            MaterialDialog(this)
                .show {
                    title(text = "服务器地址")
                    input(
                        prefill = prefs.pull("server_address", "http://183.28.157.11:30414/"),
                        allowEmpty = false
                    ) { _, sequence ->
                        prefs.push("server_address", sequence.toString())
                        makeText(context, "设置成功", Toast.LENGTH_SHORT).show()
                    }
                }
        }

        findViewById<ImageButton>(R.id.gallery).setOnClickListener {
            openAlbumWithPermissionCheck()
        }
    }

    @NeedsPermission(Manifest.permission.CAMERA)
    fun openCamera() {
        imageFile = File(externalCacheDir, "image_" + System.currentTimeMillis() + ".jpg")
        try {
            if (imageFile.exists()) {
                imageFile.delete()
            }
        } catch (ex: IOException) {
            ex.printStackTrace()
        }
        imageUri = FileProvider.getUriForFile(this, "org.cppteam.ocrbert.FileProvider", imageFile)

        val intent = Intent("android.media.action.IMAGE_CAPTURE")
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri)
        startActivityForResult(intent, TAKE_PHOTO)
    }

    @OnShowRationale(Manifest.permission.CAMERA)
    fun showRationaleForCamera() {
        makeText(this, "showRationaleForCamera", Toast.LENGTH_SHORT).show()
    }

    @OnPermissionDenied(Manifest.permission.CAMERA)
    fun onCameraDenied() {
        makeText(this, "denied", Toast.LENGTH_SHORT).show()
    }

    @OnNeverAskAgain(Manifest.permission.CAMERA)
    fun onCameraNeverAskAgain() {
        makeText(this, "never ask again", Toast.LENGTH_SHORT).show()
    }

    @NeedsPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
    fun openAlbum() {
        val intent = Intent("android.intent.action.GET_CONTENT")
        intent.type = "image/*"
        startActivityForResult(intent, PICK_PHOTO)
    }

    @OnShowRationale(Manifest.permission.READ_EXTERNAL_STORAGE)
    fun showRationaleForAlbum() {
        makeText(this, "showRationaleForCamera", Toast.LENGTH_SHORT).show()
    }

    @OnPermissionDenied(Manifest.permission.READ_EXTERNAL_STORAGE)
    fun onAlbumDenied() {
        makeText(this, "denied", Toast.LENGTH_SHORT).show()
    }

    @OnNeverAskAgain(Manifest.permission.READ_EXTERNAL_STORAGE)
    fun onAlbumNeverAskAgain() {
        makeText(this, "never ask again", Toast.LENGTH_SHORT).show()
    }

    private fun uriToFile(uri: Uri?): File {
        val res = File(this.filesDir, "image_" + System.currentTimeMillis() + ".jpg")
        val outputStream = FileOutputStream(res)
        if (uri != null) {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                outputStream.write(inputStream.readBytes())
            }
        }
        outputStream.close()
        return res
    }

    @NeedsPermission(Manifest.permission.INTERNET)
    private suspend fun postImage(imageFile: File) {
        val clientBuilder = OkHttpClient.Builder()
        clientBuilder.apply {
            clientBuilder.readTimeout(100, TimeUnit.SECONDS);
            //连接超时
            clientBuilder.connectTimeout(60, TimeUnit.SECONDS);
            //写入超时
            clientBuilder.writeTimeout(60, TimeUnit.SECONDS);
        }
        val okHttpClient = clientBuilder.build()
        val compressedImageFile = Compressor.compress(context = this, imageFile) {
            resolution(1280, 720)
            quality(80)
            format(Bitmap.CompressFormat.JPEG)
            size(2_097_152)
        }
        val image: RequestBody = compressedImageFile.asRequestBody()
        val requestBody: RequestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", compressedImageFile.path, image)
            .build()
        val request: Request = Request.Builder()
            .url(prefs.pull("server_address", "http://183.28.157.11:30414/") + "upload/")
            .post(requestBody)
            .build()

        Snackbar.make(findViewById(R.id.fab), "正在上传图片至服务器...", Snackbar.LENGTH_LONG).show()

        okHttpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Snackbar.make(
                    findViewById(R.id.fab),
                    "上传失败，请确定网络可用(Error 1)",
                    Snackbar.LENGTH_SHORT
                )
                    .show()
                Log.d("TAG", "onFailure: $e")
            }

            override fun onResponse(
                call: Call,
                response: Response
            ) {
                for ((name, value) in response.headers) {
                    println("$name: $value")
                }
                val gson = Gson()
                val serverResponse =
                    gson.fromJson(response.body?.string(), ServerResponse::class.java)
                if (serverResponse.code == "200") {
                    Log.d("TAG", "onResponse: " + serverResponse.message)
                } else {
                    makeText(
                        this@MainActivity,
                        "服务器出错，请确定服务器可用(Error 2)",
                        Toast.LENGTH_SHORT
                    ).show()
                }
                response.close()
                ResultBottomSheet(serverResponse.url, serverResponse.content).show(
                    supportFragmentManager,
                    "Result"
                )
                imageUri = null
            }
        })
    }

    @OnShowRationale(Manifest.permission.INTERNET)
    fun showRationaleForInternet() {
        makeText(this, "showRationaleForInternet", Toast.LENGTH_SHORT).show()
    }

    @OnPermissionDenied(Manifest.permission.INTERNET)
    fun onInternetDenied() {
        makeText(this, "denied", Toast.LENGTH_SHORT).show()
    }

    @OnNeverAskAgain(Manifest.permission.INTERNET)
    fun onInternetNeverAskAgain() {
        makeText(this, "never ask again", Toast.LENGTH_SHORT).show()
    }

    inner class ServerResponse(var code: String, var message: String, var content: String, var url: String)

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        onRequestPermissionsResult(requestCode, grantResults)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        when (requestCode) {
            TAKE_PHOTO -> {
                if (resultCode == RESULT_OK) {
                    CAMERA_SUCCEED = 1
                }
            }
            PICK_PHOTO -> {
                if (resultCode == RESULT_OK) {
                    imageUri = data?.data
                    GALLERY_SUCCEED = 1
                }
            }
            else -> {
                super.onActivityResult(requestCode, resultCode, data)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        Log.d("TAG", "onResume: ")
        CoroutineScope(Dispatchers.IO).launch {
            if (imageUri != null && CAMERA_SUCCEED == 1) {
                postImage(imageFile)
                CAMERA_SUCCEED = 0
            }
            if (imageUri != null && GALLERY_SUCCEED == 1) {
                postImage(uriToFile(imageUri))
                GALLERY_SUCCEED = 0
            }
        }
    }

}
package org.cppteam.ocrbert.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import com.bumptech.glide.Glide
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import kotlinx.android.synthetic.main.bottom_sheet_result.*
import org.cppteam.ocrbert.R


class ResultBottomSheet(private val resultUrl: String, private val resultText: String) :
    BottomSheetDialogFragment() {

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.bottom_sheet_result, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        Glide.with(view)
            .load(resultUrl)
            .into(image)

        result.text = resultText

        copy.setOnClickListener{
            copyToClipboard(resultText)
        }

    }

    private fun copyToClipboard(text: String) {
        val clipboardManager: ClipboardManager = context?.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboardManager.setPrimaryClip(ClipData.newPlainText(null, text))
        Toast.makeText(context, "复制成功", Toast.LENGTH_SHORT).show()
    }

}
# chatbot/forms.py
from django import forms

class MessageForm(forms.Form):
    user_input = forms.CharField(
        label='', 
        widget=forms.Textarea(attrs={
            'rows': 2, 
            'class': 'form-control', 
            'placeholder': 'Escribe tu consulta aquí...'
            }),
        required=False
    )
    image_upload = forms.ImageField(
        label="Adjuntar Imagen (Opcional)",
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control-file mt-2 mb-2'})
    )

    def clean(self):
        cleaned_data = super().clean()
        user_input_text = cleaned_data.get("user_input")
        image = cleaned_data.get("image_upload")

        if not user_input_text and not image:
            raise forms.ValidationError(
                "Debes escribir un mensaje o subir una imagen.",
                code='no_input'
            )
        return cleaned_data
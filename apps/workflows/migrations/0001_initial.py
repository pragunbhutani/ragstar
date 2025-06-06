# Generated by Django 5.2 on 2025-05-03 17:52

import django.db.models.deletion
import pgvector.django
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("knowledge_base", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Question",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("question_text", models.TextField()),
                ("answer_text", models.TextField(blank=True, null=True)),
                (
                    "question_embedding",
                    pgvector.django.VectorField(
                        blank=True,
                        dimensions=1536,
                        help_text="Embedding vector for the question text",
                        null=True,
                    ),
                ),
                ("was_useful", models.BooleanField(blank=True, null=True)),
                ("feedback", models.TextField(blank=True, null=True)),
                (
                    "feedback_embedding",
                    pgvector.django.VectorField(
                        blank=True,
                        dimensions=1536,
                        help_text="Embedding vector for the feedback text",
                        null=True,
                    ),
                ),
                (
                    "question_metadata",
                    models.JSONField(blank=True, default=dict, null=True),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("original_message_text", models.TextField(blank=True, null=True)),
                (
                    "original_message_ts",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "response_message_ts",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "original_message_embedding",
                    pgvector.django.VectorField(
                        blank=True,
                        dimensions=1536,
                        help_text="Embedding vector for the original message text",
                        null=True,
                    ),
                ),
                (
                    "response_file_message_ts",
                    models.CharField(
                        blank=True, db_index=True, max_length=50, null=True
                    ),
                ),
            ],
            options={
                "verbose_name": "Question",
                "verbose_name_plural": "Questions",
                "db_table": "questions",
            },
        ),
        migrations.CreateModel(
            name="QuestionModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("relevance_score", models.IntegerField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "model",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="knowledge_base.model",
                    ),
                ),
                (
                    "question",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="workflows.question",
                    ),
                ),
            ],
            options={
                "verbose_name": "Question Model Usage",
                "verbose_name_plural": "Question Model Usages",
                "db_table": "question_models",
                "unique_together": {("question", "model")},
            },
        ),
        migrations.AddField(
            model_name="question",
            name="models_used",
            field=models.ManyToManyField(
                related_name="questions_used_in",
                through="workflows.QuestionModel",
                to="knowledge_base.model",
            ),
        ),
    ]
